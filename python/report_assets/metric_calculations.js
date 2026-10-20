(() => {
  const popup = document.getElementById('metric-calculation-popup');
  if (!popup) return;
  const content = popup.querySelector('.calculation-content');
  const hint = popup.querySelector('.calculation-hint');
  let source = null;
  let pinned = false;
  let timer = null;
  let restoringFocus = false;
  const cancelClose = () => window.clearTimeout(timer);

  function position() {
    if (!source || popup.hidden) return;
    const anchor = source.getBoundingClientRect();
    const box = popup.getBoundingClientRect();
    const margin = 12;
    const left = Math.max(margin, Math.min(anchor.right - box.width, window.innerWidth - box.width - margin));
    const below = anchor.bottom + 8;
    const top = below + box.height <= window.innerHeight - margin ? below :
      Math.max(margin, Math.min(anchor.top - box.height - 8, window.innerHeight - box.height - margin));
    popup.style.left = `${left}px`;
    popup.style.top = `${top}px`;
  }

  function close(restoreFocus = false) {
    cancelClose();
    const old = source;
    if (old) old.setAttribute('aria-expanded', 'false');
    source = null;
    pinned = false;
    popup.hidden = true;
    if (restoreFocus && old) {
      restoringFocus = true;
      old.focus();
      restoringFocus = false;
    }
  }

  function show(button, pin = false) {
    if (restoringFocus || (pinned && !pin)) return;
    const template = document.getElementById(button.dataset.calculation);
    if (!template) return;
    cancelClose();
    if (source) source.setAttribute('aria-expanded', 'false');
    source = button;
    pinned = pin;
    content.replaceChildren(template.content.cloneNode(true));
    content.scrollTop = 0;
    hint.textContent = pin ? 'Przypięte. Esc lub × zamyka.' : 'Kliknij wynik, aby przypiąć. Esc zamyka.';
    source.setAttribute('aria-expanded', 'true');
    popup.hidden = false;
    position();
  }

  function scheduleClose() {
    cancelClose();
    if (!pinned) timer = window.setTimeout(() => {
      if (!popup.matches(':hover') && !popup.contains(document.activeElement)) close();
    }, 220);
  }

  const boundButtons = new WeakSet();
  function bindButtons() {
  document.querySelectorAll('.metric-value[data-calculation]').forEach(button => {
    if (boundButtons.has(button)) return;
    boundButtons.add(button);
    button.addEventListener('pointerenter', () => show(button));
    button.addEventListener('pointerleave', scheduleClose);
    button.addEventListener('focus', () => show(button));
    button.addEventListener('blur', scheduleClose);
    button.addEventListener('click', event => {
      if (pinned && source === button) close();
      else {
        show(button, true);
        if (event.detail === 0) content.focus();
      }
    });
  });
  }
  bindButtons();
  document.addEventListener('metric-calculations-refresh', () => {
    close();
    bindButtons();
  });
  popup.addEventListener('pointerenter', cancelClose);
  popup.addEventListener('pointerleave', scheduleClose);
  popup.addEventListener('focusout', scheduleClose);
  popup.querySelector('.calculation-close').addEventListener('click', () => close(true));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !popup.hidden) {
      event.preventDefault();
      close(true);
    }
  });
  document.addEventListener('click', event => {
    if (source && !popup.contains(event.target) && !source.contains(event.target)) close();
  });
  window.addEventListener('resize', position);
  window.addEventListener('scroll', event => {
    if (!source || popup.contains(event.target)) return;
    const rect = source.getBoundingClientRect();
    if (!pinned || rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) close();
    else position();
  }, true);
})();
