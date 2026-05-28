const placeholderLinks = document.querySelectorAll('[data-placeholder-link]');

placeholderLinks.forEach((link) => {
  link.addEventListener('click', (event) => {
    event.preventDefault();
  });
});

const dangerVideo = document.querySelector('#danger-video');
const dangerGraph = document.querySelector('#danger-graph');
const dangerSection = document.querySelector('#danger-section');

const SVG_NS = 'http://www.w3.org/2000/svg';

function createSvgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);

  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, value);
  });

  return element;
}

function getDangerPoints(data) {
  return Object.values(data.danger_by_state || {})
    .map((state) => ({
      step: Number(state.step),
      danger: Number(state.danger_score),
    }))
    .filter((point) => Number.isFinite(point.step) && Number.isFinite(point.danger))
    .sort((a, b) => a.step - b.step);
}

function renderDangerGraph(points) {
  const width = 720;
  const height = 420;
  const padding = {
    top: 24,
    right: 18,
    bottom: 38,
    left: 38,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const dangerValues = points.map((point) => point.danger);
  const meanDanger = dangerValues.reduce((sum, danger) => sum + danger, 0) / dangerValues.length;
  const rawMinDanger = Math.min(0, ...dangerValues);
  const rawMaxDanger = Math.max(...dangerValues);
  const rawDangerRange = rawMaxDanger - rawMinDanger || 1;
  const minDanger = rawMinDanger - rawDangerRange * 0.08;
  const maxDanger = rawMaxDanger + rawDangerRange * 0.08;
  const dangerRange = maxDanger - minDanger || 1;
  const minStep = points[0].step;
  const maxStep = points[points.length - 1].step;
  const stepRange = maxStep - minStep || 1;
  let pendingSeekRatio = null;
  let animationFrameId = null;
  let dangerInteractionActive = false;

  function xForRatio(ratio) {
    return padding.left + ratio * plotWidth;
  }

  function yForDanger(danger) {
    return padding.top + (1 - (danger - minDanger) / dangerRange) * plotHeight;
  }

  function ratioFromPointerEvent(event) {
    const rect = hitArea.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    return Math.max(0, Math.min(1, ratio));
  }

  function pointForRatio(ratio) {
    const targetStep = minStep + Math.max(0, Math.min(1, ratio)) * stepRange;

    return points.reduce((closest, point) => {
      const closestDistance = Math.abs(closest.step - targetStep);
      const pointDistance = Math.abs(point.step - targetStep);
      return pointDistance < closestDistance ? point : closest;
    }, points[0]);
  }

  function ratioForPoint(point) {
    return (point.step - minStep) / stepRange;
  }

  function canSeekVideo() {
    return dangerVideo && Number.isFinite(dangerVideo.duration) && dangerVideo.duration > 0;
  }

  const svg = createSvgElement('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-labelledby': 'danger-graph-title danger-graph-desc',
  });

  const title = createSvgElement('title', { id: 'danger-graph-title' });
  title.textContent = 'Dangerousness over time';
  const desc = createSvgElement('desc', { id: 'danger-graph-desc' });
  desc.textContent = 'Click the graph to seek the dangerousness video to the corresponding time.';
  svg.append(title, desc);

  const gridLines = [];
  const gridLineCount = 4;

  for (let index = 0; index <= gridLineCount; index += 1) {
    const ratio = index / gridLineCount;
    const y = padding.top + ratio * plotHeight;
    gridLines.push(createSvgElement('line', {
      class: 'danger-grid',
      x1: padding.left,
      y1: y,
      x2: padding.left + plotWidth,
      y2: y,
    }));
  }

  const bottomAxis = createSvgElement('line', {
    class: 'danger-axis',
    x1: padding.left,
    y1: padding.top + plotHeight,
    x2: padding.left + plotWidth,
    y2: padding.top + plotHeight,
  });
  const leftAxis = createSvgElement('line', {
    class: 'danger-axis',
    x1: padding.left,
    y1: padding.top,
    x2: padding.left,
    y2: padding.top + plotHeight,
  });

  const pathData = points
    .map((point, index) => {
      const ratio = (point.step - minStep) / stepRange;
      const command = index === 0 ? 'M' : 'L';
      return `${command} ${xForRatio(ratio)} ${yForDanger(point.danger)}`;
    })
    .join(' ');

  const dangerLine = createSvgElement('path', {
    class: 'danger-line',
    d: pathData,
  });
  const meanY = yForDanger(meanDanger);
  const meanLine = createSvgElement('line', {
    class: 'danger-mean-line',
    x1: padding.left,
    y1: meanY,
    x2: padding.left + plotWidth,
    y2: meanY,
  });
  const meanLabel = createSvgElement('text', {
    class: 'danger-mean-label',
    x: padding.left + plotWidth - 6,
    y: meanY - 6,
    'text-anchor': 'end',
  });
  meanLabel.textContent = 'Mean Dangerousness';

  const marker = createSvgElement('line', {
    class: 'danger-marker',
    x1: padding.left,
    y1: padding.top,
    x2: padding.left,
    y2: padding.top + plotHeight,
  });
  const hitArea = createSvgElement('rect', {
    class: 'danger-hit-area',
    x: padding.left,
    y: padding.top,
    width: plotWidth,
    height: plotHeight,
  });
  const hoverLine = createSvgElement('line', {
    class: 'danger-hover-line',
    x1: padding.left,
    y1: padding.top,
    x2: padding.left,
    y2: padding.top + plotHeight,
    visibility: 'hidden',
  });
  const hoverDot = createSvgElement('circle', {
    class: 'danger-hover-dot',
    cx: padding.left,
    cy: padding.top,
    r: 4,
    visibility: 'hidden',
  });
  const hoverLabel = createSvgElement('text', {
    class: 'danger-hover-label',
    x: padding.left + 8,
    y: padding.top + 16,
    visibility: 'hidden',
  });

  const xLabel = createSvgElement('text', {
    class: 'danger-axis-label',
    x: padding.left + plotWidth / 2,
    y: padding.top + plotHeight + 25,
    'text-anchor': 'middle',
  });
  xLabel.textContent = 'Time';

  const yLabel = createSvgElement('text', {
    class: 'danger-axis-label',
    x: 14,
    y: padding.top + plotHeight / 2,
    transform: `rotate(-90 14 ${padding.top + plotHeight / 2})`,
    'text-anchor': 'middle',
  });
  yLabel.textContent = 'Dangerousness';

  svg.append(
    ...gridLines,
    bottomAxis,
    leftAxis,
    meanLine,
    meanLabel,
    dangerLine,
    marker,
    hoverLine,
    hoverDot,
    hoverLabel,
    xLabel,
    yLabel,
    hitArea,
  );
  dangerGraph.replaceChildren(svg);

  function updateMarker(ratio) {
    const x = xForRatio(Math.max(0, Math.min(1, ratio)));
    marker.setAttribute('x1', x);
    marker.setAttribute('x2', x);
  }

  function seekVideoToRatio(ratio) {
    const boundedRatio = Math.max(0, Math.min(1, ratio));
    updateMarker(boundedRatio);
    pendingSeekRatio = boundedRatio;

    if (!canSeekVideo()) {
      return;
    }

    dangerVideo.currentTime = boundedRatio * dangerVideo.duration;
    pendingSeekRatio = null;
  }

  function seekVideoBySeconds(deltaSeconds) {
    if (!canSeekVideo()) {
      const currentRatio = pendingSeekRatio || 0;
      const fallbackStep = deltaSeconds > 0 ? 1 / Math.max(1, points.length - 1) : -1 / Math.max(1, points.length - 1);
      seekVideoToRatio(currentRatio + fallbackStep);
      return;
    }

    const nextTime = Math.max(0, Math.min(dangerVideo.duration, dangerVideo.currentTime + deltaSeconds));
    dangerVideo.currentTime = nextTime;
    updateMarker(nextTime / dangerVideo.duration);
  }

  function seekVideoByFrames(frameDelta) {
    const frameRatio = 1 / Math.max(1, points.length - 1);

    if (!canSeekVideo()) {
      seekVideoToRatio((pendingSeekRatio || 0) + frameDelta * frameRatio);
      return;
    }

    seekVideoBySeconds(frameDelta * dangerVideo.duration * frameRatio);
  }

  function updateHover(event) {
    const hoveredPoint = pointForRatio(ratioFromPointerEvent(event));
    const ratio = ratioForPoint(hoveredPoint);
    const x = xForRatio(ratio);
    const y = yForDanger(hoveredPoint.danger);
    const labelX = Math.min(x + 8, padding.left + plotWidth - 88);
    const labelY = Math.max(padding.top + 14, y - 10);

    hoverLine.setAttribute('x1', x);
    hoverLine.setAttribute('x2', x);
    hoverDot.setAttribute('cx', x);
    hoverDot.setAttribute('cy', y);
    hoverLabel.setAttribute('x', labelX);
    hoverLabel.setAttribute('y', labelY);
    hoverLabel.textContent = `danger: ${hoveredPoint.danger.toFixed(3)}`;
    hoverLine.setAttribute('visibility', 'visible');
    hoverDot.setAttribute('visibility', 'visible');
    hoverLabel.setAttribute('visibility', 'visible');
  }

  function hideHover() {
    hoverLine.setAttribute('visibility', 'hidden');
    hoverDot.setAttribute('visibility', 'hidden');
    hoverLabel.setAttribute('visibility', 'hidden');
  }

  function handleArrowScrub(event) {
    if (event.defaultPrevented) {
      return;
    }

    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      return;
    }

    event.preventDefault();
    const frameDelta = event.shiftKey ? 5 : 1;

    if (event.key === 'Home') {
      seekVideoToRatio(0);
    } else if (event.key === 'End') {
      seekVideoToRatio(1);
    } else if (event.key === 'ArrowLeft') {
      seekVideoByFrames(-frameDelta);
    } else {
      seekVideoByFrames(frameDelta);
    }
  }

  hitArea.addEventListener('pointermove', updateHover);
  hitArea.addEventListener('pointerleave', hideHover);
  dangerGraph.addEventListener('pointerenter', () => {
    dangerInteractionActive = true;
  });
  dangerSection?.addEventListener('focusin', () => {
    dangerInteractionActive = true;
  });
  dangerGraph.addEventListener('keydown', handleArrowScrub);
  document.addEventListener('pointerdown', (event) => {
    dangerInteractionActive = dangerSection?.contains(event.target) || false;
  });
  document.addEventListener('keydown', (event) => {
    if (!dangerInteractionActive) {
      return;
    }

    handleArrowScrub(event);
  }, true);

  if (dangerVideo) {
    dangerVideo.addEventListener('pointerdown', () => {
      dangerInteractionActive = true;
    });
    dangerVideo.addEventListener('keydown', handleArrowScrub);

    const updateFromVideo = () => {
      if (!canSeekVideo()) {
        updateMarker(0);
        return;
      }

      updateMarker(dangerVideo.currentTime / dangerVideo.duration);
    };
    const applyPendingSeek = () => {
      if (pendingSeekRatio !== null && canSeekVideo()) {
        dangerVideo.currentTime = pendingSeekRatio * dangerVideo.duration;
        pendingSeekRatio = null;
      }

      updateFromVideo();
    };
    const updateWhilePlaying = () => {
      updateFromVideo();

      if (!dangerVideo.paused && !dangerVideo.ended) {
        animationFrameId = window.requestAnimationFrame(updateWhilePlaying);
      }
    };

    dangerVideo.addEventListener('loadedmetadata', applyPendingSeek);
    dangerVideo.addEventListener('timeupdate', updateFromVideo);
    dangerVideo.addEventListener('seeked', updateFromVideo);
    dangerVideo.addEventListener('play', () => {
      if (animationFrameId !== null) {
        window.cancelAnimationFrame(animationFrameId);
      }

      animationFrameId = window.requestAnimationFrame(updateWhilePlaying);
    });
    dangerVideo.addEventListener('pause', () => {
      if (animationFrameId !== null) {
        window.cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }

      updateFromVideo();
    });
    updateFromVideo();
  }
}

async function initDangerGraph() {
  if (!dangerGraph) {
    return;
  }

  try {
    const response = await fetch(dangerGraph.dataset.dangerJson);

    if (!response.ok) {
      throw new Error(`Could not load ${dangerGraph.dataset.dangerJson}`);
    }

    const data = await response.json();
    const points = getDangerPoints(data);

    if (points.length === 0) {
      throw new Error('No danger scores found.');
    }

    renderDangerGraph(points);
  } catch (error) {
    dangerGraph.innerHTML = '<p class="graph-fallback">Dangerousness graph could not be loaded.</p>';
  }
}

initDangerGraph();
