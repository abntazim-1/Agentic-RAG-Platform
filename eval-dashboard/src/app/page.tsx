"use client";

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, Clock, Target, CheckCircle2, AlertTriangle, Zap, Server } from 'lucide-react';

interface Metric {
  id: string;
  query: string;
  answer: string;
  faithfulness: number | null;
  answer_relevancy: number | null;
  total_time: number | null;
  ttft: number | null;
  timestamp: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/eval/metrics');
        if (!res.ok) throw new Error('Failed to fetch metrics');
        const data = await res.json();
        setMetrics(data.reverse()); // Chronological for charts
        setError(null);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    // Poll every 5 seconds for live updates
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const validLatency = metrics.filter(m => m.total_time !== null);
  const avgLatency = validLatency.length 
    ? (validLatency.reduce((acc, curr) => acc + (curr.total_time || 0), 0) / validLatency.length).toFixed(2)
    : '0.00';

  const validFaithfulness = metrics.filter(m => m.faithfulness !== null);
  const avgFaithfulness = validFaithfulness.length
    ? (validFaithfulness.reduce((acc, curr) => acc + (curr.faithfulness || 0), 0) / validFaithfulness.length).toFixed(2)
    : '0.00';

  const chartData = metrics.map((m, index) => ({
    name: `Q${index + 1}`,
    latency: m.total_time || 0,
    ttft: m.ttft || 0,
    faithfulness: m.faithfulness ? m.faithfulness * 100 : null,
    relevancy: m.answer_relevancy ? m.answer_relevancy * 100 : null,
  }));

  if (loading && metrics.length === 0) {
    return <div className="container flex-center" style={{ height: '100vh' }}><h2>Initializing RAG Platform...</h2></div>;
  }

  return (
    <div className="container">
      <h1>RAG Observability Platform</h1>
      <p style={{ marginBottom: '2rem' }}>Live benchmark comparisons and system health evaluation.</p>

      {error && (
        <div className="card" style={{ borderLeft: '4px solid var(--accent-danger)', marginBottom: '2rem' }}>
          <p className="text-danger flex-center" style={{ justifyContent: 'flex-start', gap: '0.5rem' }}>
            <AlertTriangle size={20} /> Failed to connect to FastAPI Backend ({error}). Ensure it is running on port 8000.
          </p>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid-cols-3">
        <div className="card">
          <div className="card-header">
            <span className="card-title"><Activity size={18} className="text-accent"/> Total Queries</span>
          </div>
          <div className="card-value">{metrics.length}</div>
          <div className="card-subtitle">Last 50 recorded operations</div>
        </div>
        
        <div className="card">
          <div className="card-header">
            <span className="card-title"><Zap size={18} style={{color: '#f59e0b'}}/> Avg Latency</span>
          </div>
          <div className="card-value">{avgLatency}s</div>
          <div className="card-subtitle">End-to-end generation time</div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title"><Target size={18} className="text-success"/> Avg Faithfulness</span>
          </div>
          <div className="card-value">{avgFaithfulness}</div>
          <div className="card-subtitle">RAGAS Hallucination Score (0-1)</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid-cols-2">
        <div className="card">
          <div className="card-header">
            <span className="card-title"><Clock size={18}/> Latency & TTFT Trend (s)</span>
          </div>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" />
                <XAxis dataKey="name" stroke="var(--text-secondary)" />
                <YAxis stroke="var(--text-secondary)" />
                <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="latency" stroke="var(--accent-primary)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey="ttft" stroke="var(--accent-secondary)" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title"><CheckCircle2 size={18} className="text-success"/> Quality Metrics (%)</span>
          </div>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" />
                <XAxis dataKey="name" stroke="var(--text-secondary)" />
                <YAxis stroke="var(--text-secondary)" domain={[0, 100]} />
                <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: 'none', borderRadius: '8px' }} />
                <Bar dataKey="faithfulness" fill="var(--accent-success)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="relevancy" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Trace Table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Server size={18}/> Recent Query Traces</span>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Query</th>
                <th>Latency (s)</th>
                <th>Faithfulness</th>
                <th>Relevance</th>
              </tr>
            </thead>
            <tbody>
              {[...metrics].reverse().map((m) => (
                <tr key={m.id}>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                    {new Date(m.timestamp).toLocaleTimeString()}
                  </td>
                  <td style={{ maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {m.query}
                  </td>
                  <td>{m.total_time?.toFixed(2) || '-'}</td>
                  <td>
                    {m.faithfulness !== null ? (
                      <span className={m.faithfulness > 0.8 ? 'badge badge-success' : 'badge badge-warning'}>
                        {m.faithfulness.toFixed(2)}
                      </span>
                    ) : '-'}
                  </td>
                  <td>
                    {m.answer_relevancy !== null ? (
                      <span className={m.answer_relevancy > 0.8 ? 'badge badge-success' : 'badge badge-warning'}>
                        {m.answer_relevancy.toFixed(2)}
                      </span>
                    ) : '-'}
                  </td>
                </tr>
              ))}
              {metrics.length === 0 && (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>No queries logged yet. Chat with the RAG system to generate metrics!</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
