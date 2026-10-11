// Video Analytics Dashboard — one card per stream, fed by SSE

const statusEl = document.getElementById('status');
const streamsEl = document.getElementById('streams');
const eventLogEl = document.getElementById('event-log');
const kMaxEvents = 50;

// stream id -> the elements of its card
const cards = {};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Build a card: title, MJPEG image, stats, zone and line counters
function addCard(stream, index) {
  const card = el('div', 'card');
  card.appendChild(el('div', 'card-title', stream.id));

  const wrapper = el('div', 'video-wrapper');
  const img = el('img');
  img.src = '/video?stream=' + index;
  img.alt = stream.id;
  wrapper.appendChild(img);
  card.appendChild(wrapper);

  const stats = el('div', 'stats-bar');
  const refs = {zones: [], linesIn: [], linesOut: []};
  [['fps', 'FPS'], ['latency', 'Latency ms'], ['count', 'Objects']]
      .forEach(function(item) {
        const span = el('span', null, item[1] + ': ');
        refs[item[0]] = el('strong', null, '--');
        span.appendChild(refs[item[0]]);
        stats.appendChild(span);
      });
  card.appendChild(stats);

  const counters = el('div', 'det-list');
  stream.zones.forEach(function(name) {
    const chip = el('div', 'det-item zone', name + ': ');
    const value = el('strong', null, '0');
    chip.appendChild(value);
    refs.zones.push(value);
    counters.appendChild(chip);
  });
  stream.lines.forEach(function(name) {
    const chip = el('div', 'det-item line', name + ' in ');
    const inValue = el('strong', null, '0');
    const outValue = el('strong', null, '0');
    chip.appendChild(inValue);
    chip.appendChild(document.createTextNode(' out '));
    chip.appendChild(outValue);
    refs.linesIn.push(inValue);
    refs.linesOut.push(outValue);
    counters.appendChild(chip);
  });
  card.appendChild(counters);

  streamsEl.appendChild(card);
  cards[stream.id] = refs;
}

function logEvent(event) {
  const time = new Date(event.timestamp * 1000).toLocaleTimeString();
  let text = time + '  ' + event.stream + '  ' + event.type + '  ' +
             event.name + '  ' + event.class + ' #' + event.track_id;
  if (event.direction) text += '  ' + event.direction;
  if (event.dwell_s !== undefined) text += '  ' + event.dwell_s + ' s';
  eventLogEl.insertBefore(el('div', 'event ' + event.type, text),
                          eventLogEl.firstChild);
  while (eventLogEl.children.length > kMaxEvents) {
    eventLogEl.removeChild(eventLogEl.lastChild);
  }
}

// Update one card from an SSE message
function update(data) {
  const refs = cards[data.stream];
  if (!refs) return;
  refs.fps.textContent = data.fps.toFixed(1);
  refs.latency.textContent = data.latency_ms.toFixed(0);
  refs.count.textContent = data.detections.length;
  (data.zones || []).forEach(function(n, i) {
    if (refs.zones[i]) refs.zones[i].textContent = n;
  });
  (data.lines_in || []).forEach(function(n, i) {
    if (refs.linesIn[i]) refs.linesIn[i].textContent = n;
  });
  (data.lines_out || []).forEach(function(n, i) {
    if (refs.linesOut[i]) refs.linesOut[i].textContent = n;
  });
  (data.events || []).forEach(logEvent);
}

// Connect to SSE stream with auto-reconnect
function connectSSE() {
  var source = new EventSource('/events');

  source.onopen = function() {
    statusEl.textContent = 'Connected';
    statusEl.className = 'status connected';
  };

  source.onmessage = function(event) {
    try {
      update(JSON.parse(event.data));
    } catch (e) {
      console.warn('Failed to parse SSE data:', e);
    }
  };

  source.onerror = function() {
    statusEl.textContent = 'Disconnected';
    statusEl.className = 'status disconnected';
    source.close();
    setTimeout(connectSSE, 2000);
  };
}

fetch('/streams')
    .then(function(response) { return response.json(); })
    .then(function(streams) {
      streams.forEach(addCard);
      connectSSE();
    });
