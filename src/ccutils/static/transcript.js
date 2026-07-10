document.querySelectorAll('time[data-timestamp]').forEach(function(el) {
    const timestamp = el.getAttribute('data-timestamp');
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    if (isToday) { el.textContent = timeStr; }
    else { el.textContent = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + timeStr; }
});
document.querySelectorAll('pre.json').forEach(function(el) {
    const text = el.textContent;
    el.textContent = '';
    const tokenRegex = /("([^"]+)":)|(: "([^"]*)")|(: (\d+))|(: (true|false|null))|([^":\s]+|[:"\s]+)/g;
    let match;
    while ((match = tokenRegex.exec(text)) !== null) {
        const fullToken = match[0];
        if (match[1]) {
            const key = match[2];
            const span = document.createElement('span');
            span.style.color = '#ce93d8';
            span.textContent = '"' + key + '"';
            el.appendChild(span);
            el.appendChild(document.createTextNode(':'));
        } else if (match[3]) {
            const val = match[4];
            el.appendChild(document.createTextNode(': '));
            const span = document.createElement('span');
            span.style.color = '#81d4fa';
            span.textContent = '"' + val + '"';
            el.appendChild(span);
        } else if (match[5]) {
            const val = match[6];
            el.appendChild(document.createTextNode(': '));
            const span = document.createElement('span');
            span.style.color = '#ffcc80';
            span.textContent = val;
            el.appendChild(span);
        } else if (match[7]) {
            const val = match[8];
            el.appendChild(document.createTextNode(': '));
            const span = document.createElement('span');
            span.style.color = '#f48fb1';
            span.textContent = val;
            el.appendChild(span);
        } else {
            el.appendChild(document.createTextNode(fullToken));
        }
    }
});
document.querySelectorAll('.truncatable').forEach(function(wrapper) {
    const content = wrapper.querySelector('.truncatable-content');
    const btn = wrapper.querySelector('.expand-btn');
    if (content.scrollHeight > 250) {
        wrapper.classList.add('truncated');
        btn.addEventListener('click', function() {
            if (wrapper.classList.contains('truncated')) { wrapper.classList.remove('truncated'); wrapper.classList.add('expanded'); btn.textContent = 'Show less'; }
            else { wrapper.classList.remove('expanded'); wrapper.classList.add('truncated'); btn.textContent = 'Show more'; }
        });
    }
});
