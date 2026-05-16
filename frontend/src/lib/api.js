const API_URL = 'http://127.0.0.1:8020';

export async function analyzeEcg(testId, filterType) {
    const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            test_id: testId,
            filter_type: filterType.toLowerCase(),
        }),
    });

    if (!res.ok) {
        let detail = `HTTP ${res.status}`;

        try {
            const err = await res.json();
            detail = err.detail || detail;
        } catch { }

        throw new Error(detail);
    }

    return await res.json();
}

export async function fetchTests() {
    const res = await fetch(`${API_URL}/tests`);

    if (!res.ok) {
        throw new Error('Failed to fetch tests');
    }

    return await res.json();
}