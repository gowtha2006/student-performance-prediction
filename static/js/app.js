document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    if (window.lucide) {
        lucide.createIcons();
    }

    // Global State
    let currentMetrics = null;
    let radarChartInstance = null;

    // Feature Sliders Configuration
    const sliders = [
        { id: 'previous_score', valId: 'val_previous_score' },
        { id: 'study_hours', valId: 'val_study_hours' },
        { id: 'attendance', valId: 'val_attendance' },
        { id: 'assignment_score', valId: 'val_assignment_score' },
        { id: 'internal_marks', valId: 'val_internal_marks' },
        { id: 'sleep_hours', valId: 'val_sleep_hours' },
        { id: 'participation', valId: 'val_participation' }
    ];

    // ==========================================================================
    // 1. Sliders Synchronization & Real-Time Input Events
    // ==========================================================================
    sliders.forEach(({ id, valId }) => {
        const sliderEl = document.getElementById(id);
        const displayEl = document.getElementById(valId);
        if (sliderEl && displayEl) {
            sliderEl.addEventListener('input', (e) => {
                displayEl.textContent = parseFloat(e.target.value).toFixed(id === 'participation' ? 0 : 1);
                // Debounced auto prediction
                clearTimeout(window.predictTimer);
                window.predictTimer = setTimeout(triggerPrediction, 150);
            });
        }
    });

    document.getElementById('parental_support')?.addEventListener('change', triggerPrediction);
    document.getElementById('extracurricular')?.addEventListener('change', triggerPrediction);
    document.getElementById('btn-predict')?.addEventListener('click', triggerPrediction);

    // ==========================================================================
    // 2. Quick Profile Presets
    // ==========================================================================
    document.getElementById('preset-topper')?.addEventListener('click', () => {
        setFormValues({
            study_hours: 7.5,
            attendance: 96.0,
            previous_score: 88.0,
            assignment_score: 92.0,
            internal_marks: 22.5,
            sleep_hours: 7.5,
            participation: 9,
            parental_support: 3,
            extracurricular: 1
        });
        triggerPrediction();
    });

    document.getElementById('preset-average')?.addEventListener('click', () => {
        setFormValues({
            study_hours: 4.5,
            attendance: 85.0,
            previous_score: 68.0,
            assignment_score: 65.0,
            internal_marks: 16.0,
            sleep_hours: 7.0,
            participation: 6,
            parental_support: 2,
            extracurricular: 1
        });
        triggerPrediction();
    });

    document.getElementById('preset-struggling')?.addEventListener('click', () => {
        setFormValues({
            study_hours: 1.8,
            attendance: 62.0,
            previous_score: 44.0,
            assignment_score: 45.0,
            internal_marks: 9.5,
            sleep_hours: 5.0,
            participation: 3,
            parental_support: 1,
            extracurricular: 0
        });
        triggerPrediction();
    });

    function setFormValues(vals) {
        Object.keys(vals).forEach(key => {
            const el = document.getElementById(key);
            const valDisplay = document.getElementById(`val_${key}`);
            if (el) {
                el.value = vals[key];
            }
            if (valDisplay) {
                valDisplay.textContent = parseFloat(vals[key]).toFixed(key === 'participation' ? 0 : 1);
            }
        });
    }

    function getFormData() {
        return {
            study_hours: parseFloat(document.getElementById('study_hours').value),
            attendance: parseFloat(document.getElementById('attendance').value),
            previous_score: parseFloat(document.getElementById('previous_score').value),
            assignment_score: parseFloat(document.getElementById('assignment_score').value),
            internal_marks: parseFloat(document.getElementById('internal_marks').value),
            sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
            participation: parseInt(document.getElementById('participation').value),
            parental_support: parseInt(document.getElementById('parental_support').value),
            extracurricular: parseInt(document.getElementById('extracurricular').value),
            model_type: 'best'
        };
    }

    // ==========================================================================
    // 3. Score Prediction API & Visual Updates
    // ==========================================================================
    async function triggerPrediction() {
        const formData = getFormData();
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) throw new Error('Prediction API error');
            const data = await response.json();

            if (data.success) {
                updatePredictionUI(data);
                updateRadarChart(formData, data.comparison);
            }
        } catch (err) {
            console.error('Prediction request failed:', err);
        }
    }

    function updatePredictionUI(data) {
        const score = data.predicted_score;
        const perf = data.performance;

        // Animated Score Counter
        animateCounter('res-score', score);
        
        // Update Ring Progress (circumference = 2 * PI * 76 ≈ 477.52)
        const ring = document.getElementById('score-ring');
        const circumference = 477.52;
        const offset = circumference - (score / 100) * circumference;
        if (ring) {
            ring.style.strokeDashoffset = offset;
            ring.style.stroke = perf.color;
        }

        // Update Badges & Details
        const badge = document.getElementById('res-badge');
        if (badge) {
            badge.textContent = perf.level;
            badge.className = `badge ${perf.badge_class}`;
        }

        document.getElementById('res-grade').textContent = perf.grade;
        document.getElementById('res-grade').style.color = perf.color;
        document.getElementById('res-level').textContent = perf.level;
        document.getElementById('res-summary').textContent = perf.summary;

        // Render Recommendations
        renderRecommendations(data.recommendations);
    }

    function animateCounter(id, targetVal) {
        const el = document.getElementById(id);
        if (!el) return;
        const startVal = parseFloat(el.textContent) || 0;
        const duration = 350;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = startVal + (targetVal - startVal) * progress;
            el.textContent = current.toFixed(1);
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                el.textContent = targetVal.toFixed(1);
            }
        }
        requestAnimationFrame(update);
    }

    function renderRecommendations(recs) {
        const list = document.getElementById('recommendations-list');
        if (!list) return;
        list.innerHTML = '';

        recs.forEach(rec => {
            const item = document.createElement('div');
            item.className = `rec-card rec-${rec.type}`;
            item.innerHTML = `
                <div class="rec-icon">
                    <i data-lucide="${rec.icon}"></i>
                </div>
                <div class="rec-content">
                    <h5>${rec.title}</h5>
                    <p>${rec.text}</p>
                </div>
            `;
            list.appendChild(item);
        });

        if (window.lucide) {
            lucide.createIcons();
        }
    }

    // ==========================================================================
    // 4. Radar Chart: Student Profile vs Cohort Average
    // ==========================================================================
    function updateRadarChart(formData, comparison) {
        const canvas = document.getElementById('radarChart');
        if (!canvas) return;

        const labels = ['Study Hrs', 'Attendance', 'Prev Exam', 'Assignment', 'Internal', 'Sleep', 'Participation'];
        
        // Scale values to 0-100 for visual uniformity
        const userData = [
            formData.study_hours * 10,
            formData.attendance,
            formData.previous_score,
            formData.assignment_score,
            (formData.internal_marks / 25) * 100,
            (formData.sleep_hours / 10) * 100,
            formData.participation * 10
        ];

        const avgData = comparison ? [
            comparison.study_hours.cohort_avg * 10,
            comparison.attendance.cohort_avg,
            comparison.previous_score.cohort_avg,
            comparison.assignment_score.cohort_avg,
            (comparison.internal_marks.cohort_avg / 25) * 100,
            (comparison.sleep_hours.cohort_avg / 10) * 100,
            comparison.participation.cohort_avg * 10
        ] : [45, 86, 68, 58, 62, 68, 63];

        if (radarChartInstance) {
            radarChartInstance.data.datasets[0].data = userData;
            radarChartInstance.data.datasets[1].data = avgData;
            radarChartInstance.update();
            return;
        }

        radarChartInstance = new Chart(canvas, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Student Profile',
                        data: userData,
                        backgroundColor: 'rgba(99, 102, 241, 0.25)',
                        borderColor: '#6366f1',
                        pointBackgroundColor: '#6366f1',
                        pointBorderColor: '#fff',
                        borderWidth: 2
                    },
                    {
                        label: 'Cohort Average',
                        data: avgData,
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderColor: '#10b981',
                        borderDash: [4, 4],
                        pointBackgroundColor: '#10b981',
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#cbd5e1', font: { size: 10, family: 'Plus Jakarta Sans' } },
                        ticks: { display: false, min: 0, max: 100 }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { size: 11 } }
                    }
                }
            }
        });
    }

    // ==========================================================================
    // 5. Load Backend Metrics for Header Stats
    // ==========================================================================
    async function loadMetrics() {
        try {
            const res = await fetch('/api/metrics');
            if (!res.ok) return;
            currentMetrics = await res.json();

            if (currentMetrics.best_model) {
                document.getElementById('top-best-model').textContent = currentMetrics.best_model;
            }
            if (currentMetrics.metrics && currentMetrics.metrics[currentMetrics.best_model]) {
                const best = currentMetrics.metrics[currentMetrics.best_model];
                document.getElementById('top-r2').textContent = `${(best.r2 * 100).toFixed(1)}%`;
            }
            if (currentMetrics.dataset_stats) {
                document.getElementById('top-samples').textContent = `${currentMetrics.dataset_stats.total_students} Students`;
            }
        } catch (e) {
            console.error('Failed to load metrics:', e);
        }
    }

    // Initialize
    loadMetrics();
    triggerPrediction();
});
