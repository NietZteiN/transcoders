import fs from "fs";
const html=fs.readFileSync("nla_results.html","utf8");
const jsAll=html.slice(html.lastIndexOf("<script>",html.indexOf("const DATA = window.__NLA__")));
const grab=n=>{const i=jsAll.indexOf("function "+n+"(");let d=0,j=jsAll.indexOf("{",i);
 for(let k=j;k<jsAll.length;k++){if(jsAll[k]==="{")d++;else if(jsAll[k]==="}"){d--;if(!d)return jsAll.slice(i,k+1);}}};
const esc=s=>(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const MARK_CAP=120; const NEW=eval("("+grab("cotHTML")+")");
function OLD(c){const reply=c.model_reply||"";
 const marks=c.reads.map((r,i)=>({i,a:r.anchor})).filter(x=>x.a&&x.a.in_reply).sort((x,y)=>x.a.rs-y.a.rs);
 let out="",cur=0;
 for(const m of marks){const s=Math.max(cur,m.a.rs),e=Math.max(s,Math.min(m.a.re,reply.length));
  if(s>=reply.length)break; out+=esc(reply.slice(cur,s));
  out+='<mark data-r="'+m.i+'">'+esc(reply.slice(s,e))+'</mark>'; cur=e;} return out+esc(reply.slice(cur));}

// which characters does reading i actually end up highlighted under?
function coverage(h){
  const cov={}; let plain="", re=/<mark data-r="([\d,]+)"[^>]*>(.*?)<\/mark>|([^<]+)/gs, m;
  while((m=re.exec(h))){
    if(m[1]!==undefined){ const txt=m[2].replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&quot;/g,'"');
      m[1].split(",").forEach(n=>{cov[n]=cov[n]||[]; cov[n].push([plain.length, plain.length+txt.length]);});
      plain+=txt;
    } else plain+=m[3].replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&quot;/g,'"');
  }
  return cov;
}
const reply="The function computes fibfib(4) = 0+1+0 = 1 and then continues to the next step.";
const synth={model_reply:reply,task_key:"s",reads:[0,1,2,3,4].map(k=>({rt_cos:.8,
  anchor:{in_reply:true,rs:22+2*k,re:22+2*k+6}}))};
for(const [lab,fn] of [["OLD",OLD],["NEW",NEW]]){
  const cov=coverage(fn(synth));
  console.log(`\n${lab}:`);
  let wrong=0;
  for(let i=0;i<5;i++){
    const want=[22+2*i, 22+2*i+6];
    const got=cov[i]||[];
    const lo=got.length?Math.min(...got.map(g=>g[0])):null, hi=got.length?Math.max(...got.map(g=>g[1])):null;
    const ok = lo===want[0] && hi===want[1];
    if(!ok) wrong++;
    console.log(`  read ${i}: true span [${want}] -> highlighted [${lo},${hi}] ${ok?"":"  << WRONG: shows "+JSON.stringify(reply.slice(lo,hi))+" instead of "+JSON.stringify(reply.slice(...want))}`);
  }
  console.log(`  misplaced highlights: ${wrong}/5`);
}
