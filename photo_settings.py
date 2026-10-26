"""Validated, serializable processing settings. No private paths are stored here."""
import math, re

DEFAULTS = dict(alignment_mode='warn', disagreement=1.5, minimum_correlation=.90,
    maximum_shift=16., region_mode='auto', regions=[], kernel='triangle', support=1.,
    fallback_support=1.5, exposure=0., white_balance='camera', temperature=6500., tint=0.,
    sharpening=0., sharpening_radius=.8, denoise=0., encoding='srgb', border=0,
    filename='Pixelshift-{first}-{count}frames')
RANGES = dict(disagreement=(.05,16),minimum_correlation=(.5,.99999),maximum_shift=(1,64),
    support=(.5,2),fallback_support=(1.1,2),exposure=(-5,5),temperature=(2000,15000),
    tint=(-100,100),sharpening=(0,2),sharpening_radius=(.3,3),denoise=(0,100),border=(0,128))
CHOICES=dict(alignment_mode=('warn','strict'),region_mode=('auto','manual'),kernel=('triangle','cosine'),
    white_balance=('camera','custom'),encoding=('srgb','linear'))

def validate(values=None):
    result={**DEFAULTS,**(values or {})}
    result={k:result[k] for k in DEFAULTS}
    for key,(low,high) in RANGES.items():
        value=float(result[key])
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{key} must be between {low} and {high}.')
        result[key]=int(value) if key=='border' else value
    for key,choices in CHOICES.items():
        if result[key] not in choices:raise ValueError(f'Unsupported {key}.')
    regions=result['regions']
    if not isinstance(regions,list) or len(regions)>20:raise ValueError('Use at most 20 alignment areas.')
    normalized=[]
    for r in regions:
        if r.get('kind') not in ('include','exclude'):raise ValueError('Invalid alignment area.')
        rectangle={k:float(r[k]) for k in ('x','y','w','h')}
        if any(not math.isfinite(v) for v in rectangle.values()):raise ValueError('Invalid alignment coordinates.')
        if min(rectangle.values())<0 or rectangle['w']<.01 or rectangle['h']<.01 or rectangle['x']+rectangle['w']>1.0001 or rectangle['y']+rectangle['h']>1.0001:raise ValueError('Alignment area extends beyond the image.')
        normalized.append({**rectangle,'kind':r['kind']})
    result['regions']=normalized
    if result['region_mode']=='manual' and not any(r['kind']=='include' for r in normalized):raise ValueError('Draw at least one Include area for manual alignment.')
    name=str(result['filename']).strip()
    if not name or len(name)>100 or re.search(r'[/\\\x00-\x1f]',name):raise ValueError('Use a filename template without directory separators.')
    if re.search(r'\{(?!first\}|count\}|date\})',name):raise ValueError('Filename placeholders: {first}, {count}, {date}.')
    try:name.format(first='frame',count=16,date='20260101')
    except (ValueError,KeyError):raise ValueError('Invalid filename template.')
    result['filename']=name
    return result
