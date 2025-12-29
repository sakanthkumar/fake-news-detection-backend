// Theme Management
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeButton();
}

function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
    }
    updateThemeButton();
}

function updateThemeButton() {
    const btn = document.querySelector('[onclick="toggleTheme()"]');
    const isDark = document.body.classList.contains('dark-mode');
    btn.innerHTML = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}

// Prediction Functions
async function predict(event) {
    event.preventDefault();
    const headline = document.getElementById('headline_input').value.trim();
    if (!headline) return;

    showLoading('predict_result');

    try {
        const token = localStorage.getItem('token');
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ headline })
        });
        const data = await res.json();
        
        if (data.error) {
            showError('predict_result', data.error);
            return;
        }

        showResult('predict_result', data);
    } catch (err) {
        showError('predict_result', 'Analysis failed. Please try again.');
    }
}

async function scrapeUrl(event) {
    event.preventDefault();
    const url = document.getElementById('url_input').value.trim();
    if (!url) return;

    showLoading('scrape_result');

    try {
        const token = localStorage.getItem('token');
        const res = await fetch('/api/scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ url })
        });
        const data = await res.json();

        if (data.error) {
            showError('scrape_result', data.error);
            return;
        }

        showResult('scrape_result', data);
    } catch (err) {
        showError('scrape_result', 'URL analysis failed. Please try again.');
    }
}

// Helper Functions
function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary"></div>
            <p class="mt-2">Analyzing...</p>
        </div>
    `;
}

function showError(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="alert alert-danger">
            ${message}
        </div>
    `;
}

function showResult(elementId, data) {
    const confidencePercent = (data.confidence <= 1 ? data.confidence * 100 : data.confidence).toFixed(1);
    const credibilityClass = data.credibility >= 60 ? 'bg-success' : 
                            data.credibility >= 30 ? 'bg-warning' : 'bg-danger';

    document.getElementById(elementId).innerHTML = `
        <div class="card results-card fade-in">
            <div class="card-body">
                <div class="mb-3">
                    <strong>Headline:</strong> ${data.headline}
                </div>
                
                ${data.language ? `
                    <div class="mb-3">
                        <strong>Language:</strong> 
                        <span class="badge bg-secondary">${data.language}</span>
                    </div>
                ` : ''}

                <div class="mb-3">
                    <strong>Verdict:</strong>
                    <span class="badge ms-2 ${data.prediction === 'Real' ? 'bg-success' : 'bg-danger'}">
                        ${data.prediction}
                    </span>
                    <span class="ms-2">
                        (${confidencePercent}% confidence)
                    </span>
                </div>

                <div class="mb-3">
                    <strong>Credibility Score:</strong>
                    <div class="progress mt-2">
                        <div class="progress-bar ${credibilityClass}"
                             role="progressbar"
                             style="width: ${data.credibility}%"
                             aria-valuenow="${data.credibility}"
                             aria-valuemin="0"
                             aria-valuemax="100">
                            ${data.credibility}%
                        </div>
                    </div>
                </div>

                <div class="mb-3">
                    <strong>Matched Sources:</strong><br>
                    <div class="mt-2">
                        ${(data.matched_sources || []).length > 0 
                            ? data.matched_sources.map(source => 
                                `<span class="badge bg-info text-dark source-badge">${source}</span>`
                              ).join('')
                            : '<small class="text-muted">No trusted-source matches found</small>'
                        }
                    </div>
                </div>

                <button onclick="explainResult('${data.headline}')" class="btn btn-outline-info btn-sm">
                    <i class="fas fa-info-circle me-2"></i>
                    Explain This Result
                </button>
            </div>
        </div>
    `;
}

async function explainResult(headline) {
    try {
        const token = localStorage.getItem('token');
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ headline })
        });
        const data = await res.json();

        if (data.error) {
            document.getElementById('explanation').innerHTML = `
                <div class="alert alert-danger">
                    ${data.error}
                </div>
            `;
            return;
        }

        document.getElementById('explanation').innerHTML = `
            <div class="card fade-in">
                <div class="card-body">
                    <h5 class="card-title">
                        <i class="fas fa-lightbulb text-warning me-2"></i>
                        Why This Prediction?
                    </h5>
                    <div class="mt-3">
                        ${data.explanation.map(item => `
                            <div class="word-impact mb-2">
                                <span>"${item.word}"</span>
                                <span class="badge ${item.impact === 'supports' ? 'bg-success' : 'bg-danger'}">
                                    ${item.impact === 'supports' ? 'Supports' : 'Contradicts'}
                                </span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    } catch (err) {
        document.getElementById('explanation').innerHTML = `
            <div class="alert alert-danger">
                Could not get explanation. Please try again.
            </div>
        `;
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('theme');
    window.location.href = '/login';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    if (!localStorage.getItem('token')) {
        window.location.href = '/login';
        return;
    }
    initTheme();
});