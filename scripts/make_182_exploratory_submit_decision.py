import json
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
SUBMIT_AGENT = Path("E:/Jitter/.tmp/submit-agent")
TARGET_CANDIDATE = "177b_w010_top1_guard"
TARGET_ZIP = "submissions/177b_w010_top1_guard_repack_checked/result.zip"


def load_json(rel):
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def zip_root_ok(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted(z.namelist())
    return names == ["dataset1.csv", "dataset2.csv"], names


def validate(zip_rel):
    proc = subprocess.run(
        ["python", "validate_result_zip.py", "--zip", zip_rel],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=240,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def write_submit_script():
    SUBMIT_AGENT.mkdir(parents=True, exist_ok=True)
    js_path = SUBMIT_AGENT / "submit_track1_177b_exploratory_monitor.js"
    js = r"""
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const USER_DATA_DIR = process.env.LOCALAPPDATA + '/Microsoft/Edge/User Data';
const URL = 'https://www.educoder.net/competitions/Jittor-7?login=puf6ylfeq&websiteName=educoder';
const REPO = 'E:/Jitter/track1_aggressive_wsl';
const ZIP_PATH = `${REPO}/submissions/177b_w010_top1_guard_repack_checked/result.zip`;
const DECISION_JSON = `${REPO}/analysis/182_exploratory_submit_decision.json`;
const OUT_DIR = 'E:/Jitter/.tmp/submit-agent';
const RESULT_JSON = `${OUT_DIR}/submit_track1_177b_result.json`;
const WAIT_AFTER_CONFIRM_MS = Number(process.env.WAIT_AFTER_CONFIRM_MS || 100000);
const TEXT = { submitResult: '\u63d0\u4ea4\u7ed3\u679c', track1: '\u8d5b\u9053\u4e00\uff1a\u57fa\u4e8e\u56fe\u5b66\u4e60\u7684\u52a8\u6001\u63a8\u8350\u4efb\u52a1', evalCount: '\u63d0\u4ea4\u8bc4\u6d4b\u6b21\u6570', confirm: '\u786e\u5b9a' };
const RE = { humanGate: /\u9a8c\u8bc1\u7801|\u77ed\u4fe1|\u4eba\u673a|\u4e8c\u6b21|\u6ed1\u5757|\u5b89\u5168\u9a8c\u8bc1/, login: /\u767b\u5f55\s*\/\s*\u6ce8\u518c|\u767b\u5f55|\u6ce8\u518c/, registered: /\u5df2\u62a5\u540d/, submittedStatuses: /\u8ba1\u7b97\u4e2d|\u5f85\u8ba1\u7b97|\u5b8c\u6210|\u6210\u529f|\u6392\u961f/ };
function nowIso(){return new Date().toISOString();}
function writeResult(r){fs.mkdirSync(OUT_DIR,{recursive:true});fs.writeFileSync(RESULT_JSON,JSON.stringify(r,null,2),'utf8');}
function preflight(){
  if(!ZIP_PATH.includes('177b_w010_top1_guard_repack_checked')) throw new Error(`unsafe target ${ZIP_PATH}`);
  const d=JSON.parse(fs.readFileSync(DECISION_JSON,'utf8'));
  if(d.decision !== 'submit_exploratory_177b') throw new Error(`decision is ${d.decision}, not submit_exploratory_177b`);
  if(d.submit_target !== 'submissions/177b_w010_top1_guard_repack_checked/result.zip') throw new Error(`target mismatch ${d.submit_target}`);
  if(d.preflight.auto_eval_decision !== 'exploratory_submit') throw new Error(`bad auto_eval ${d.preflight.auto_eval_decision}`);
  if(d.preflight.dataset1_mad_vs_143 !== 0) throw new Error('dataset1 changed');
  if(d.preflight.dataset2_mad_vs_143 < 0.01 || d.preflight.dataset2_mad_vs_143 > 0.04) throw new Error('mad gate failed');
  if(d.preflight.top1_change_vs_143 > 0.01) throw new Error('top1 gate failed');
  if(d.preflight.changed_rows_vs_143 < 50000) throw new Error('changed rows gate failed');
  const root=execFileSync('python',['-c',`import zipfile\np=r'''${ZIP_PATH}'''\nwith zipfile.ZipFile(p) as z:\n names=sorted(z.namelist())\nassert names==['dataset1.csv','dataset2.csv'],names\nprint(names)`],{encoding:'utf8'}).trim();
  const val=execFileSync('python',['validate_result_zip.py','--zip',ZIP_PATH],{cwd:REPO,encoding:'utf8'}).trim();
  return {decision:d,zipRoot:root,validate:val};
}
function killEdge(){try{execFileSync('powershell',['-NoProfile','-Command',"Get-Process msedge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"],{encoding:'utf8'});return{ok:true};}catch(e){return{ok:false,error:String(e).slice(0,800)}}}
(async()=>{
  const result={ok:false,targetZip:ZIP_PATH,resultJson:RESULT_JSON,startedAt:nowIso(),waitAfterConfirmMs:WAIT_AFTER_CONFIRM_MS,attempts:[]};
  try{result.preflight=preflight();console.log(`AUTO SUBMIT TARGET:\n${ZIP_PATH}`);}catch(e){result.blocked='preflight_failed';result.error=String(e).slice(0,1600);result.finishedAt=nowIso();writeResult(result);console.log(JSON.stringify(result,null,2));return;}
  result.edgeCleanup=killEdge(); await new Promise(r=>setTimeout(r,2500));
  let context;
  try{context=await chromium.launchPersistentContext(USER_DATA_DIR,{executablePath:EDGE,headless:false,args:['--profile-directory=Default'],viewport:{width:1440,height:1000},timeout:30000});}
  catch(e){result.blocked='browser_profile_locked_or_launch_failed';result.error=String(e).slice(0,2000);result.finishedAt=nowIso();writeResult(result);console.log(JSON.stringify(result,null,2));return;}
  const page=context.pages()[0]||await context.newPage();
  async function shot(x){await page.screenshot({path:path.join(OUT_DIR,`${x}.png`),fullPage:true}).catch(()=>{});}
  async function text(){return await page.locator('body').innerText().catch(()=>'');}
  async function noGate(phase){if(RE.humanGate.test(await text())) throw new Error(`human_verification_required:${phase}`);}
  async function clickText(t){const b=await page.evaluate((txt)=>{const vis=el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return r.width>0&&r.height>0&&s.display!=='none'&&s.visibility!=='hidden'};return [...document.querySelectorAll('button,a,span,div')].map(el=>{const r=el.getBoundingClientRect();return{text:(el.innerText||'').trim().replace(/\s+/g,' '),x:r.x,y:r.y,w:r.width,h:r.height,visible:vis(el)}}).filter(x=>x.visible&&x.text===txt&&x.w>0&&x.w<520&&x.h>0&&x.h<140).sort((a,b)=>(a.y-b.y)||(a.x-b.x))[0]||null;},t);if(!b)throw new Error(`text not found:${t}`);await page.mouse.click(b.x+b.w/2,b.y+b.h/2);await page.waitForTimeout(1800);}
  async function selectTrack1(){await page.goto(URL,{waitUntil:'domcontentloaded',timeout:60000});await page.waitForTimeout(5000);await noGate('initial');const tx=await text();if(RE.login.test(tx)&&!RE.registered.test(tx))throw new Error('not_logged_in');await clickText(TEXT.submitResult);await page.mouse.click(735,594);await page.waitForTimeout(2500);const st=await text();if(!st.includes(TEXT.track1)||!st.includes(TEXT.evalCount))throw new Error('track1_submit_page_not_confirmed');}
  async function latest(){return await page.evaluate(()=>{const rows=[...document.querySelectorAll('tr')].map(tr=>[...tr.querySelectorAll('th,td')].map(td=>(td.innerText||'').trim().replace(/\s+/g,' '))).filter(c=>/^20\d{14,}$/.test((c[0]||'').replace(/\s/g,'')));if(!rows.length)return null;const c=rows[0];return{requestId:(c[0]||'').replace(/\s/g,''),attachment:c[1]||'',submitter:c[2]||'',submittedAt:c[3]||'',status:c[4]||'',score:c[5]||'',info:c[6]||'',raw:c};});}
  async function upload(){await page.mouse.click(1240,594);await page.waitForTimeout(1500);await page.locator('input[type=file]').first().waitFor({state:'attached',timeout:10000});await page.locator('input[type=file]').first().setInputFiles(ZIP_PATH);await page.waitForFunction((confirmText)=>{const btn=[...document.querySelectorAll('.ant-modal button')].find(b=>(b.innerText||'').trim()===confirmText);if(!btn)return false;return !btn.disabled&&!btn.getAttribute('aria-disabled')&&!String(btn.className||'').includes('loading');},TEXT.confirm,{timeout:60000});await page.locator('.ant-modal button').filter({hasText:TEXT.confirm}).last().click({timeout:10000,force:true});}
  try{const a={attemptNo:1,startedAt:nowIso()};result.attempts.push(a);await selectTrack1();await shot('submit_177b_before');const before=await latest();a.beforeLatestRow=before;await upload();a.confirmClickedAt=nowIso();await shot('submit_177b_confirm_clicked');await page.waitForTimeout(WAIT_AFTER_CONFIRM_MS);await selectTrack1();await shot('submit_177b_after_100s');const after=await latest();a.after100sInspect=after;const newReq=!!(after&&after.requestId&&(!before||after.requestId!==before.requestId));result.ok=!!(newReq&&RE.submittedStatuses.test(after.status));result.status=result.ok?'submitted_or_scored':'submit_status_unknown_no_retry';result.latestRow=after;}
  catch(e){result.blocked='script_error_or_platform_gate';result.error=String(e).slice(0,2000);await shot('submit_177b_error').catch(()=>{});}
  finally{result.finishedAt=nowIso();writeResult(result);console.log(JSON.stringify(result,null,2));await context.close().catch(()=>{});}
})();
"""
    js_path.write_text(js.strip() + "\n", encoding="utf-8")
    return str(js_path)


def main():
    ANALYSIS.mkdir(exist_ok=True)
    r181 = load_json("analysis/181_exploratory_submit_policy_fix.json")
    selected = r181.get("selected_exploratory_candidate") or {}
    summary = load_json("outputs/website_submission_177b_w010_top1_guard/summary.json")
    zip_path = ROOT / TARGET_ZIP
    root_pass = False
    names = []
    val_pass = False
    val_output = ""
    reasons = []
    if selected.get("candidate") != TARGET_CANDIDATE:
        reasons.append(f"181 selected {selected.get('candidate')}, expected {TARGET_CANDIDATE}")
    if not zip_path.exists():
        reasons.append(f"zip not found: {TARGET_ZIP}")
    else:
        root_pass, names = zip_root_ok(zip_path)
        val_pass, val_output = validate(TARGET_ZIP)
        if not root_pass:
            reasons.append(f"zip root mismatch: {names}")
        if not val_pass:
            reasons.append("validate_result_zip failed")
    checks = {
        "candidate": TARGET_CANDIDATE,
        "dataset1_mad_vs_143": summary.get("dataset1_mad_vs_143"),
        "dataset2_mad_vs_143": summary.get("dataset2_mad_vs_143"),
        "changed_rows_vs_143": summary.get("changed_rows_vs_143"),
        "top1_change_vs_143": summary.get("top1_change_vs_143"),
        "real_data_derived_signal": summary.get("real_data_derived_signal"),
        "not_submission_output_only": summary.get("not_submission_output_only"),
        "not_score_shape_only": summary.get("not_score_shape_only"),
        "not_lgbm_takeover": summary.get("not_lgbm_takeover"),
        "not_router_expansion": summary.get("not_router_expansion"),
        "validate_pass": val_pass,
        "zip_root_pass": root_pass,
        "zip_names": names,
        "auto_eval_decision": "exploratory_submit",
    }
    if checks["dataset1_mad_vs_143"] != 0:
        reasons.append("dataset1_mad_vs_143 != 0")
    mad_value = 0.0 if checks["dataset2_mad_vs_143"] is None else float(checks["dataset2_mad_vs_143"])
    top1_value = 1.0 if checks["top1_change_vs_143"] is None else float(checks["top1_change_vs_143"])
    changed_value = 0 if checks["changed_rows_vs_143"] is None else int(checks["changed_rows_vs_143"])
    if not (0.01 <= mad_value <= 0.04):
        reasons.append("dataset2_mad_vs_143 not in [0.01,0.04]")
    if top1_value > 0.01:
        reasons.append("top1_change_vs_143 > 0.01")
    if changed_value < 50000:
        reasons.append("changed_rows_vs_143 < 50000")
    for key in ["real_data_derived_signal", "not_submission_output_only", "not_score_shape_only", "not_lgbm_takeover", "not_router_expansion"]:
        if not checks[key]:
            reasons.append(f"{key} is false")
    submit_script = write_submit_script()
    decision = "submit_exploratory_177b" if not reasons else "do_not_submit"
    online_status = "pending_submit" if decision.startswith("submit") else "not_submitted"
    out = {
        "id": 182,
        "name": "exploratory_submit_decision",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": decision,
        "submit_target": TARGET_ZIP if decision.startswith("submit") else "",
        "submit_mode": "exploratory_submit",
        "preflight": checks,
        "preflight_reasons": reasons,
        "submit_script": submit_script,
        "auto_submit_attempted": False,
        "online_status": online_status,
        "online_score": None,
        "request_id": "",
        "validate_output": val_output,
    }
    (ANALYSIS / "182_exploratory_submit_decision.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# 182_exploratory_submit_decision",
        "",
        f"- decision: {decision}",
        f"- submit_target: `{TARGET_ZIP if decision.startswith('submit') else '-'}`",
        f"- submit_script: `{submit_script}`",
        f"- dataset2_mad_vs_143: {checks['dataset2_mad_vs_143']}",
        f"- changed_rows_vs_143: {checks['changed_rows_vs_143']}",
        f"- top1_change_vs_143: {checks['top1_change_vs_143']}",
        f"- validate_pass: {val_pass}",
        f"- zip_root_pass: {root_pass}",
        f"- reasons: {reasons if reasons else 'all exploratory preflight checks passed'}",
        "",
        "该决策只允许提交 177b_w010_top1_guard，一个包一次；不允许降级提交 raw 168 或其他 177 候选。",
    ]
    (ANALYSIS / "182_exploratory_submit_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "submit_target": out["submit_target"], "submit_script": submit_script}, ensure_ascii=False))


if __name__ == "__main__":
    main()
