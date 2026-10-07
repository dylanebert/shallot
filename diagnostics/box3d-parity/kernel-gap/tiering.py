"""Map V8 compilation diagnostics to WASM names (no timing claims)."""
import pathlib,re,json,sys
b=pathlib.Path(sys.argv[1]).read_bytes(); at=8

def leb():
    global at
    n=shift=0
    while True:
        c=b[at]; at+=1; n|=(c&127)<<shift
        if c<128:return n
        shift+=7

def string():
    global at
    n=leb(); s=b[at:at+n].decode(); at+=n; return s
names={}
while at<len(b):
    section=b[at]; at+=1; size=leb(); end=at+size
    if section==0 and string()=='name':
        while at<end:
            sub=b[at]; at+=1; size=leb(); subend=at+size
            if sub==1:
                for _ in range(leb()):
                    index=leb(); names[index]=string()
            at=subend
    at=end
out={}
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    match=re.search(r'#(\d+) using (\w+).*bodysize (\d+) codesize (\d+)',line)
    if not match:continue
    index,tier,size,code=match.groups(); name=names.get(int(index),index)
    if any(key in name for key in ['contact_block','joint5solve','support_vertex_wide','tree5query','clip_polygon','collide_hulls']):
        out.setdefault(name,[]).append({'tier':tier,'wasm_body_bytes':int(size),'machine_code_bytes':int(code)})
pathlib.Path('diagnostics/box3d-parity/kernel-gap/tiering.json').write_text(json.dumps(out,indent=2)+'\n')
