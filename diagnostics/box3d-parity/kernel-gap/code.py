"""Summarize named optimized WAT functions; WAT comes from wasm-dis."""
import pathlib, re, hashlib, json, subprocess, sys, tempfile
labels=['pairs-before','pairs-after','rain-before','rain-after']
if len(sys.argv)!=5: raise SystemExit('usage: python3 diagnostics/box3d-parity/kernel-gap/code.py PAIRS_BEFORE_ROOT PAIRS_AFTER_ROOT RAIN_BEFORE_ROOT RAIN_AFTER_ROOT')
temporary=tempfile.TemporaryDirectory(prefix='shallot-wat-')
work=pathlib.Path(temporary.name)
for tag,root in zip(labels,sys.argv[1:]):
    for kind,relative in [('single','target/shallot_physics.opt.wasm'),('shared','crates/physics/target-shared/shallot_physics.opt.wasm')]:
        subprocess.run(['wasm-dis',str(pathlib.Path(root)/relative),'-o',str(work/(tag+'-'+kind+'.wat'))],check=True)
out={}
for tag in labels:
    out[tag]={}
    for kind in ['single','shared']:
        text=(work/(tag+'-'+kind+'.wat')).read_text()
        functions={}
        for m in re.finditer(r'^ \(func (\$\S+)',text,re.M):
            name=m.group(1)
            if not any(s in name for s in ['tree5query','pairwork9run_query','pairworkNt','5arena13contact_block','5joint5solve','5solve7run_job']): continue
            start=m.start(); end=start; depth=0
            while end<len(text):
                c=text[end]; end+=1
                if c=='(': depth+=1
                elif c==')':
                    depth-=1
                    if depth==0: break
            body=text[start:end]
            normalized=re.sub(r'Cs[A-Za-z0-9_]+?_15shallot_physics','CRATE',body)
            opcodes=re.findall(r'\(([A-Za-z][A-Za-z0-9_.]*)\b',body)
            relocation_normalized=re.sub(r'(i32\.const )(-?\d+)', lambda m: m.group(1)+('ADDRESS' if abs(int(m.group(2)))>65536 else m.group(2)),normalized)
            relocation_normalized=re.sub(r'offset=(\d+)', lambda m: 'offset='+('ADDRESS' if int(m.group(1))>65536 else m.group(1)),relocation_normalized)
            key=re.sub(r'Cs[A-Za-z0-9_]+?_15shallot_physics','CRATE',name)
            functions[key]={'sha256':hashlib.sha256(normalized.encode()).hexdigest(),'opcode_sha256':hashlib.sha256(' '.join(opcodes).encode()).hexdigest(),'relocation_normalized_sha256':hashlib.sha256(relocation_normalized.encode()).hexdigest(),'bytes_wat':len(body),'memory_fill':body.count('memory.fill'),'loads':len(re.findall(r'\b(?:i32|i64|f32|f64|v128)\.load',body)),'stores':len(re.findall(r'\b(?:i32|i64|f32|f64|v128)\.store',body)),'calls':body.count('(call ')}
        out[tag][kind]=functions
pathlib.Path('diagnostics/box3d-parity/kernel-gap/generated-code.json').write_text(json.dumps(out,indent=2)+'\n')
temporary.cleanup()
