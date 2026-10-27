"""Optional direct libheif decoding through its public C ABI.

Buffers remain owned by libheif for the lifetime of the context manager.
No generated PNG, compiler, or additional Python package is required.
"""
import ctypes as C
import ctypes.util
import os,shutil,sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import numpy as np

class Unavailable(RuntimeError):pass
class HeifError(ValueError):pass
class Error(C.Structure):
    _fields_=[('code',C.c_int),('subcode',C.c_int),('message',C.c_char_p)]

def candidates():
    if os.environ.get('PIXELSHIFT_LIBHEIF'):yield os.environ['PIXELSHIFT_LIBHEIF']
    found=C.util.find_library('heif')
    if found:yield found
    for directory in ('/opt/homebrew/lib','/usr/local/lib','/usr/lib','/usr/lib/x86_64-linux-gnu','/usr/lib/aarch64-linux-gnu'):
        for name in ('libheif.dylib','libheif.so.1','libheif.so'):yield str(Path(directory)/name)
    # Also recognize relocatable CLI installations, including bundled runtimes.
    exe=shutil.which('heif-convert')
    if exe:
        for ancestor in list(Path(exe).resolve().parents)[:5]:
            for relative in ('lib','native/libheif/libheif/lib'):
                directory=ancestor/relative
                if directory.is_dir():
                    for pattern in ('libheif*.dylib','libheif.so*'):yield from map(str,sorted(directory.glob(pattern)))

@lru_cache(maxsize=1)
def library():
    for path in dict.fromkeys(candidates()):
        try:lib=C.CDLL(path)
        except OSError:continue
        def bind(name,args,result):
            f=getattr(lib,name);f.argtypes=args;f.restype=result
        try:
            bind('heif_context_alloc',[],C.c_void_p)
            bind('heif_context_free',[C.c_void_p],None)
            bind('heif_context_read_from_file',[C.c_void_p,C.c_char_p,C.c_void_p],Error)
            bind('heif_context_get_primary_image_handle',[C.c_void_p,C.POINTER(C.c_void_p)],Error)
            bind('heif_decode_image',[C.c_void_p,C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_void_p],Error)
            bind('heif_image_get_plane_readonly',[C.c_void_p,C.c_int,C.POINTER(C.c_int)],C.POINTER(C.c_uint8))
            for name in ('heif_image_get_width','heif_image_get_height','heif_image_get_bits_per_pixel_range'):
                bind(name,[C.c_void_p,C.c_int],C.c_int)
            bind('heif_image_handle_release',[C.c_void_p],None)
            bind('heif_image_release',[C.c_void_p],None)
            bind('heif_context_set_max_decoding_threads',[C.c_void_p,C.c_int],None)
            lib._pixelshift_path=path
            return lib
        except AttributeError:continue
    raise Unavailable('Direct libheif library unavailable; using the installed heif-convert fallback.')

def check(error):
    if error.code:raise HeifError((error.message or b'HEIF decoding failed').decode('utf-8',errors='replace'))

@contextmanager
def pixels(path):
    lib=library();context=lib.heif_context_alloc();handle=C.c_void_p();image=C.c_void_p()
    if not context:raise MemoryError('Could not allocate a HEIF decoder.')
    try:
        # Limit tile concurrency: the codec already performs its own parallel work.
        lib.heif_context_set_max_decoding_threads(context,2)
        check(lib.heif_context_read_from_file(context,os.fsencode(path),None))
        check(lib.heif_context_get_primary_image_handle(context,C.byref(handle)))
        chroma=14 if sys.byteorder=='little' else 12  # interleaved native-endian 16-bit RGB
        check(lib.heif_decode_image(handle,C.byref(image),1,chroma,None))
        width=lib.heif_image_get_width(image,10);height=lib.heif_image_get_height(image,10)
        bits=lib.heif_image_get_bits_per_pixel_range(image,10);stride=C.c_int()
        ptr=lib.heif_image_get_plane_readonly(image,10,C.byref(stride))
        if not ptr or width<=0 or height<=0 or not 8<=bits<=16 or stride.value<width*6 or stride.value%2:
            raise HeifError('Unexpected decoded HEIF pixel layout.')
        buffer=np.ctypeslib.as_array(ptr,shape=(height*stride.value,))
        array=buffer.view(np.uint16).reshape(height,stride.value//2)[:,:width*3].reshape(height,width,3)
        array.flags.writeable=False
        yield array,bits
    finally:
        if image:lib.heif_image_release(image)
        if handle:lib.heif_image_handle_release(handle)
        lib.heif_context_free(context)
