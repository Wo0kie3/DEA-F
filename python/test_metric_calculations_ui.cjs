// Unit-test popup event logic without a browser or external DOM dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Element {
  constructor() { this.listeners = {}; this.attrs = {}; this.style = {}; this.children = []; this.hidden = true; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  fire(type, event = {}) { for (const fn of this.listeners[type] || []) fn({ target: this, ...event }); }
  setAttribute(key, value) { this.attrs[key] = value; }
  contains(node) { return this === node || this.children.some(child => child.contains(node)); }
  getBoundingClientRect() { return this.rect || { top: 100, bottom: 125, left: 200, right: 240, width: 40, height: 25 }; }
  matches() { return this.hover || false; }
  replaceChildren(child) { this.rendered = child; }
  focus() { document.activeElement = this; this.fire('focus'); }
}
const document = new Element();
const window = new Element();
window.innerWidth = 800;
window.innerHeight = 700;
const timers = new Map();
let nextTimer = 0;
window.setTimeout = fn => { timers.set(++nextTimer, fn); return nextTimer; };
window.clearTimeout = id => timers.delete(id);
function flush() { for (const fn of [...timers.values()]) fn(); timers.clear(); }
const popup = new Element(), content = new Element(), hint = new Element(), close = new Element();
popup.children = [content, hint, close];
popup.rect = { width: 580, height: 400 };
popup.querySelector = selector => ({ '.calculation-content': content, '.calculation-hint': hint, '.calculation-close': close })[selector];
const buttons = [new Element(), new Element()];
buttons.forEach((button, i) => { button.dataset = { calculation: `template-${i}` }; });
const nodes = { 'metric-calculation-popup': popup };
buttons.forEach((_, i) => { nodes[`template-${i}`] = { content: { cloneNode: () => `calculation ${i}` } }; });
document.getElementById = id => nodes[id];
document.querySelectorAll = () => buttons;
document.activeElement = null;
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'report_assets/metric_calculations.js'), 'utf8'), { document, window });
buttons[0].fire('pointerenter');
assert.equal(popup.hidden, false);
assert.equal(content.rendered, 'calculation 0');
assert.equal(buttons[0].attrs['aria-expanded'], 'true');
assert.ok(parseFloat(popup.style.left) >= 12);
assert.ok(parseFloat(popup.style.top) >= 12);
buttons[0].fire('pointerleave');
popup.hover = true;
popup.fire('pointerenter');
flush();
assert.equal(popup.hidden, false);
popup.hover = false;
popup.fire('pointerleave');
flush();
assert.equal(popup.hidden, true);
buttons[0].focus();
assert.equal(popup.hidden, false);
buttons[0].fire('click');
buttons[0].fire('pointerleave');
flush();
assert.equal(popup.hidden, false);
buttons[1].fire('pointerenter');
assert.equal(content.rendered, 'calculation 0');
buttons[1].fire('click');
assert.equal(content.rendered, 'calculation 1');
document.fire('keydown', { key: 'Escape', preventDefault() {} });
assert.equal(popup.hidden, true);
assert.equal(document.activeElement, buttons[1]);
buttons[0].fire('click');
close.fire('click');
assert.equal(popup.hidden, true);
assert.equal(document.activeElement, buttons[0]);
buttons[0].fire('click');
document.fire('click', { target: new Element() });
assert.equal(popup.hidden, true);
buttons[0].fire('pointerenter');
window.fire('scroll');
assert.equal(popup.hidden, true);
buttons[0].fire('click', { detail: 0 });
assert.equal(document.activeElement, content);
document.fire('keydown', { key: 'Escape', preventDefault() {} });
assert.equal(popup.hidden, true);
assert.equal(document.activeElement, buttons[0]);
console.log('Popup hover, focus, pin, switch, close, Escape and scroll tests passed.');
const dynamic = new Element();
dynamic.dataset = {calculation:'template-0'};
buttons.push(dynamic);
document.fire('metric-calculations-refresh');
dynamic.fire('pointerenter');
assert.equal(content.rendered,'calculation 0');
dynamic.fire('click');
assert.equal(popup.hidden,false);
document.fire('metric-calculations-refresh');
assert.equal(popup.hidden,true);
assert.equal(dynamic.listeners.click.length,1);
console.log('Dynamically inserted metric buttons refresh without duplicate handlers.');
