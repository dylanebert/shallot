"""Add windowed worker CPU sampling to an isolated diagnostic archive."""
import pathlib,re,sys
root=pathlib.Path(sys.argv[1]); p=root/'src/standard/physics/kernel/pool.ts'; t=p.read_text()
t=re.sub(r'const CTL_WORDS = (\d+);',lambda m:'const CTL_WORDS = '+str(int(m.group(1))+1)+';',t)
t='let diagnosticControl: Int32Array | undefined;\nexport function diagnosticWorkerProfile(on: number) { if (diagnosticControl) Atomics.store(diagnosticControl, diagnosticControl.length-1,on); }\n'+t
t=t.replace('const ctlView = new Int32Array(ctl);','const ctlView = new Int32Array(ctl); diagnosticControl = ctlView;')
t=t.replace('    let seen = 0;','    let diagnosticSession; let diagnosticRunning = false;\n    if (typeof require === "function") diagnosticSession = new (require("node:inspector").Session)();\n    let seen = 0;')
t=t.replace('        try {','''        const diagnosticFlag = Atomics.load(ctl,ctl.length-1);
        if (diagnosticSession && diagnosticFlag === 1 && !diagnosticRunning) {
            diagnosticSession.connect(); diagnosticSession.post("Profiler.enable");
            diagnosticSession.post("Profiler.setSamplingInterval", {interval:100});
            diagnosticSession.post("Profiler.start"); diagnosticRunning = true;
        }
        if (diagnosticSession && diagnosticFlag === 2 && diagnosticRunning) {
            diagnosticSession.post("Profiler.stop", (err,result) => { if(err) throw err; console.log("WP "+d.index+" "+JSON.stringify(result.profile)); });
            diagnosticSession.disconnect(); diagnosticRunning = false;
        }
        try {''',1)
p.write_text(t)
p=root/'diagnostics/box3d-parity/scenes.ts'; t=p.read_text(); t='import { diagnosticWorkerProfile } from "../../src/standard/physics/kernel/pool";\n'+t
t=t.replace('if (i === cpuFrom) cpu?.startCpu();','if (i === cpuFrom) { diagnosticWorkerProfile(1); cpu?.startCpu(); }')
t=t.replace('console.log(lines.join("\\n"));','if (cpu && threads > 1) { diagnosticWorkerProfile(2); w.step(f(1/60),4); }\nconsole.log(lines.join("\\n"));')
p.write_text(t)
