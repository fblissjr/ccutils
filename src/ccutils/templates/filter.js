// Client-side filter for a self-contained transcript.
//
// Every listener is attached with addEventListener. There are no inline
// on*= handlers anywhere in the output: the CSP pins this script by sha256
// hash, and a hash covers a <script> BLOCK but cannot cover an attribute --
// permitting attributes would require 'unsafe-hashes', which would permit
// any inline handler, including one buried in a transcript we are rendering.
//
// Matching runs against data-search only, never the rendered DOM. The two
// would diverge the first time styling changed what is displayed.
(function () {
  var q = document.getElementById('q');
  var shown = document.getElementById('shown');
  var none = document.getElementById('no-match');
  var messages = [].slice.call(document.querySelectorAll('.message'));
  var prompts = [].slice.call(document.querySelectorAll('.prompt-list a'));

  if (!q) { return; }

  function apply() {
    var terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    var n = 0;
    messages.forEach(function (el) {
      var hay = el.getAttribute('data-search') || '';
      // Every term must match, so two words narrow rather than widen.
      var ok = terms.every(function (t) { return hay.indexOf(t) !== -1; });
      el.classList.toggle('hidden', !ok);
      if (ok) { n++; }
    });
    prompts.forEach(function (el) {
      var hay = (el.textContent || '').toLowerCase();
      var ok = terms.every(function (t) { return hay.indexOf(t) !== -1; });
      el.parentNode.classList.toggle('hidden', !ok);
    });
    if (none) { none.classList.toggle('hidden', n !== 0 || !messages.length); }
    if (shown) {
      shown.textContent = terms.length
        ? 'showing ' + n + ' of ' + messages.length
        : '';
    }
  }

  q.addEventListener('input', apply);

  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && document.activeElement !== q) {
      e.preventDefault();
      q.focus();
    }
    if (e.key === 'Escape' && document.activeElement === q) {
      q.value = '';
      apply();
    }
  });
})();
