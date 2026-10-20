const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');

class Element {
  constructor() { this.listeners = {}; this.attrs = {}; this.hidden = false; this.innerHTML = ''; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  fire(type) { return this.listeners[type]({target:this}); }
  setAttribute(name, value) { this.attrs[name] = value; }
}
function packed(id) {
  return zlib.gzipSync(JSON.stringify({path_id:id,figure:{data:[{name:id}],layout:{title:id}},
    stages_html:`stages ${id}`,metrics_html:`metrics ${id}`})).toString('base64');
}

async function main() {
  const criterion = new Element(), status = new Element(), button = new Element();
  const plot = new Element(), stages = new Element(), metrics = new Element(), panel = new Element();
  const fields = {'.explorer-criterion':criterion,'.explorer-status':status,'.explorer-load':button,
    '.explorer-plot':plot,'.explorer-stages':stages,'.explorer-metrics':metrics};
  const base = {direction:'lower_better',selected_value:1,finite_paths:2,tested_paths:3,excluded_paths:1,
    tie_count:1,constant:false,description:'criterion'};
  fields['.explorer-data'] = {textContent:JSON.stringify({default_path:'a',bundles:{a:packed('a'),b:packed('b')},
    choices:[{...base,metric:'tc',path_id:'a'},{...base,metric:'pym',direction:'higher_better',path_id:'b'},
      {...base,metric:'rr',direction:'higher_better',path_id:'a',constant:true,tie_count:2}]})};
  panel.querySelector = selector => fields[selector];
  let refreshes = 0;
  const document = {querySelectorAll:()=>[panel],dispatchEvent:()=>{refreshes++;}};
  let initial = 0, updates = 0;
  const Plotly = {
    newPlot:async (node,data)=>{initial++;node.data=data;},
    react:async (node,data)=>{updates++;node.data=data;}
  };
  const context = {document,Plotly,Blob,Response,DecompressionStream,Uint8Array,atob,Event};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'report_assets/metric_path_explorer.js'),'utf8'),context);
  criterion.value = 'default';
  await button.fire('click');
  assert.equal(initial,1);
  assert.equal(plot.data[0].name,'a');
  assert.equal(stages.innerHTML,'stages a');
  criterion.value = 'pym';
  await criterion.fire('change');
  assert.equal(plot.data[0].name,'b');
  assert.equal(metrics.innerHTML,'metrics b');
  assert.ok(status.textContent.includes('maksimum'));
  criterion.value = 'rr';
  await criterion.fire('change');
  assert.ok(status.textContent.includes('nie różnicuje'));
  assert.ok(status.textContent.includes('2 ścieżek'));
  criterion.value = 'tc';
  const old = criterion.fire('change');
  criterion.value = 'pym';
  const latest = criterion.fire('change');
  await Promise.all([old,latest]);
  assert.equal(plot.data[0].name,'b');
  assert.equal(stages.innerHTML,'stages b');
  assert.equal(panel.attrs['aria-busy'],'false');
  assert.equal(button.hidden,true);
  assert.ok(updates>=3 && refreshes>=4);
  console.log('Metric switch, maximum direction, ties, synchronized tables, rapid changes and popup refresh passed.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
