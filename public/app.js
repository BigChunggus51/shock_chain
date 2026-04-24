let inventoryChartInstance = null;
let orderChartInstance = null;

// Chart.js defaults for dark theme
Chart.defaults.color = '#94a1b2';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';

async function init() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        
        const policySelect = document.getElementById('policySelect');
        const scenarioSelect = document.getElementById('scenarioSelect');
        
        // Setup options (using existing HTML options for now to keep labels nice,
        // but we could populate dynamically if needed)
        
        document.getElementById('runBtn').addEventListener('click', runSimulation);
        
        // Run initial simulation
        runSimulation();
    } catch (e) {
        console.error("Failed to load config", e);
    }
}

async function runSimulation() {
    const btn = document.getElementById('runBtn');
    btn.textContent = 'Simulating...';
    btn.disabled = true;
    
    const policy = document.getElementById('policySelect').value;
    const scenario = document.getElementById('scenarioSelect').value;
    
    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ policy, scenario })
        });
        
        if (!res.ok) {
            const err = await res.json();
            alert("Error: " + err.detail);
            return;
        }
        
        const data = await res.json();
        updateDashboard(data.metrics);
        
    } catch (e) {
        console.error(e);
        alert("Failed to run simulation.");
    } finally {
        btn.textContent = 'Simulate';
        btn.disabled = false;
    }
}

function updateDashboard(metrics) {
    const steps = metrics.map(m => m.step);
    
    // Stats
    const totalReward = metrics.reduce((sum, m) => sum + m.reward, 0);
    const stockoutDays = metrics.filter(m => m['retailer/backlog'] > 0).length;
    
    const orders = metrics.map(m => m['factory/order_qty']);
    const demand = metrics.map(m => m['retailer/order_qty']);
    const varOrders = getVariance(orders);
    const varDemand = getVariance(demand) || 1;
    const bullwhip = varOrders / varDemand;
    
    document.getElementById('rewardVal').textContent = totalReward.toFixed(0);
    document.getElementById('stockoutVal').textContent = `${stockoutDays}/365`;
    document.getElementById('bullwhipVal').textContent = bullwhip.toFixed(2);
    
    // Inventory Chart
    const ctxInv = document.getElementById('inventoryChart').getContext('2d');
    if (inventoryChartInstance) inventoryChartInstance.destroy();
    
    inventoryChartInstance = new Chart(ctxInv, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [
                {
                    label: 'Retailer Inventory',
                    data: metrics.map(m => m['retailer/inventory']),
                    borderColor: '#00d2ff',
                    backgroundColor: 'rgba(0, 210, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Retailer Backlog (Negative)',
                    data: metrics.map(m => -m['retailer/backlog']),
                    borderColor: '#ff4b4b',
                    backgroundColor: 'rgba(255, 75, 75, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false }
        }
    });
    
    // Order Chart (Bullwhip)
    const ctxOrd = document.getElementById('orderChart').getContext('2d');
    if (orderChartInstance) orderChartInstance.destroy();
    
    orderChartInstance = new Chart(ctxOrd, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [
                {
                    label: 'Retailer Orders (End Demand)',
                    data: metrics.map(m => m['retailer/order_qty']),
                    borderColor: '#4CAF50',
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'Warehouse Orders',
                    data: metrics.map(m => m['warehouse/order_qty']),
                    borderColor: '#FF9800',
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'Factory Orders',
                    data: metrics.map(m => m['factory/order_qty']),
                    borderColor: '#9C27B0',
                    tension: 0.2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false }
        }
    });
}

function getVariance(arr) {
    if (!arr.length) return 0;
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    return arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
}

// Start
document.addEventListener('DOMContentLoaded', init);
