(() => {
  async function unpack(encoded) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('Ten wykres wymaga przeglądarki obsługującej DecompressionStream (gzip).');
    }
    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }

  function mount(panel) {
    const payload = JSON.parse(panel.querySelector('.explorer-data').textContent);
    const criterion = panel.querySelector('.explorer-criterion');
    const status = panel.querySelector('.explorer-status');
    const button = panel.querySelector('.explorer-load');
    const plot = panel.querySelector('.explorer-plot');
    const stages = panel.querySelector('.explorer-stages');
    const metrics = panel.querySelector('.explorer-metrics');
    const cache = new Map();
    let version = 0;
    let renderQueue = Promise.resolve();
    let loaded = false;

    async function update() {
      const request = ++version;
      const choice = payload.choices.find(row => row.metric === criterion.value);
      const id = criterion.value === 'default' ? payload.default_path : choice?.path_id;
      if (!id) {
        status.textContent = 'Brak skończonych wartości tej metryki. Nie wybieramy ścieżki.';
        return;
      }
      status.textContent = 'Wczytywanie ścieżki ' + id + '...';
      panel.setAttribute('aria-busy', 'true');
      try {
        if (!cache.has(id)) cache.set(id, unpack(payload.bundles[id]));
        const bundle = await cache.get(id);
        if (request !== version) return;
        renderQueue = renderQueue.catch(() => {}).then(async () => {
          if (request !== version) return;
          const config = { responsive: true, displaylogo: false, scrollZoom: true };
          if (plot.data) await Plotly.react(plot, bundle.figure.data, bundle.figure.layout, config);
          else await Plotly.newPlot(plot, bundle.figure.data, bundle.figure.layout, config);
          if (request !== version) return;
          stages.innerHTML = bundle.stages_html;
          metrics.innerHTML = bundle.metrics_html;
          document.dispatchEvent(new Event('metric-calculations-refresh'));
          if (!choice) {
            status.textContent = `Ścieżka ${id}. Domyślny wybór leksykograficzny z raportu.`;
          } else {
            const direction = choice.direction === 'lower_better' ? 'minimum' : 'maksimum';
            const value = Number(choice.selected_value).toPrecision(7);
            status.textContent = `${choice.metric.toUpperCase()}: ${direction}, wynik ${value}. Ścieżka ${id}. ` +
              `Skończone wyniki: ${choice.finite_paths}/${choice.tested_paths}; wyłączone: ${choice.excluded_paths}. ` +
              `Remis w tolerancji 1e-9: ${choice.tie_count} ścieżek; reprezentant według TC, MSC, ID. ` +
              (choice.constant ? 'Ta metryka nie różnicuje ścieżek o skończonej wartości. ' : '') + choice.description;
          }
          loaded = true;
          button.hidden = true;
        });
        await renderQueue;
      } catch (error) {
        if (request === version) {
          status.textContent = 'Nie udało się wczytać wykresu: ' + error.message;
          button.hidden = false;
        }
      } finally {
        if (request === version) panel.setAttribute('aria-busy', 'false');
      }
    }

    criterion.addEventListener('change', update);
    button.addEventListener('click', update);
    return { load: () => { if (!loaded) return update(); } };
  }

  const panels = document.querySelectorAll('.path-explorer');
  const controllers = new Map();
  panels.forEach(panel => controllers.set(panel, mount(panel)));
  if (typeof IntersectionObserver !== 'undefined') {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        observer.unobserve(entry.target);
        controllers.get(entry.target).load();
      }
    }, { rootMargin: '200px' });
    panels.forEach(panel => observer.observe(panel));
  }
})();
