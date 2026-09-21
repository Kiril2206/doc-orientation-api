"""Generate a local, paginated uprightness review page with manifest export."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    key = hashlib.sha256(manifest.read_bytes()).hexdigest()
    html = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Проверка ориентации обучающих страниц</title>
<style>body{font:15px system-ui;margin:24px;background:#f4f6f8}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
article{background:white;padding:14px;border:1px solid #ccd4dc;border-radius:8px}
img{width:100%;height:240px;object-fit:contain}button{margin:4px;padding:8px;cursor:pointer}
header{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:1}p{overflow-wrap:anywhere}
</style><header><h1>Проверьте, где у документа верх</h1>
<p>Открывайте страницу в полном размере при сомнениях. Исправьте поворот и отметьте «Верно».
Пустые, нечитаемые и неоднозначные страницы исключайте. Прогресс сохраняется в этом браузере.
Кнопка экспорта скачает manifest.reviewed.jsonl: сохраните его рядом с исходным manifest.jsonl.</p>
<button id="prev">Назад</button><button id="next">Далее</button>
<button id="export">Скачать проверенный манифест</button><span id="status"></span></header>
<div id="grid"></div><script>
const original=__DATA__, key="orientation-review-__KEY__";
let rows=original, page=0;
try { const saved=JSON.parse(localStorage.getItem(key)); if(saved&&saved.length===rows.length)rows=saved; } catch {}
const grid=document.querySelector("#grid"), status=document.querySelector("#status");
function save(){try{localStorage.setItem(key,JSON.stringify(rows));}catch{}render();}
function render(){
 grid.replaceChildren();
 const verified=rows.filter(r=>r.upright_verified||r.exclude).length;
 status.textContent="Проверено "+verified+"/"+rows.length+"; страница "+(page+1);
 rows.slice(page*24,(page+1)*24).forEach((r,n)=>{
  const item=document.createElement("article"), p=document.createElement("p");
  p.textContent=(page*24+n+1)+". "+r.source+" / "+r.split+" / "+r.group_id;
  const link=document.createElement("a");link.href=r.path;link.target="_blank";link.rel="noopener";
  const img=document.createElement("img");img.src=r.path;img.loading="lazy";
  img.style.transform="rotate("+(r.correction_cw||0)+"deg)";link.append(img);
  const rotate=document.createElement("button");rotate.textContent="Повернуть 90°";
  rotate.onclick=()=>{r.correction_cw=((r.correction_cw||0)+90)%360;r.upright_verified=false;save();};
  const verify=document.createElement("button");verify.textContent=r.upright_verified?"✓ Верно":"Верно";
  verify.onclick=()=>{r.upright_verified=!r.upright_verified;r.exclude=false;save();};
  const exclude=document.createElement("button");exclude.textContent=r.exclude?"Исключена — вернуть":"Исключить";
  exclude.onclick=()=>{r.exclude=!r.exclude;save();};
  item.append(p,link,rotate,verify,exclude);grid.append(item);
 });
 document.querySelector("#prev").disabled=page===0;
 document.querySelector("#next").disabled=(page+1)*24>=rows.length;
}
document.querySelector("#prev").onclick=()=>{page--;render();};
document.querySelector("#next").onclick=()=>{page++;render();};
document.querySelector("#export").onclick=()=>{
 const blob=new Blob([rows.map(r=>JSON.stringify(r)).join("\\n")+"\\n"],{type:"application/x-ndjson"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download="manifest.reviewed.jsonl";a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
};
render();</script></html>'''
    html = html.replace("__DATA__", data).replace("__KEY__", key)
    output = manifest.with_name("review.html")
    output.write_text(html, encoding="utf-8")
    print(f"Open locally: {output}")


if __name__ == "__main__":
    main()
