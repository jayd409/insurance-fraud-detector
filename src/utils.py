import base64, os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='muted')

def to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def save_html(charts, title, kpis=None, path='outputs/dashboard.html'):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    kpi_html = ''
    if kpis:
        kpi_html = '<div class="kpis">' + ''.join(
            f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k,v in kpis) + '</div>'
    cards = ''.join(
        f'<div class="card"><p>{n}</p><img src="data:image/png;base64,{to_b64(f)}"/></div>'
        for n,f in charts)
    html = f"""<!DOCTYPE html><html><head><title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f6f9;padding:24px;color:#222}}
h1{{color:#1e3a5f;border-left:4px solid #3b82f6;padding-left:10px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}
.kpi{{background:#3b82f6;color:#fff;border-radius:8px;padding:14px 20px;text-align:center}}
.kpi b{{display:block;font-size:1.4rem}} .kpi span{{font-size:.8rem;opacity:.85}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}}
.card{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.card p{{font-weight:600;color:#1e3a5f;margin:0 0 8px}} .card img{{width:100%}}
footer{{margin-top:24px;color:#aaa;font-size:.8rem}}
</style></head><body><h1>{title}</h1>{kpi_html}<div class="grid">{cards}</div>
<footer>Jay Desai | jayd409@gmail.com</footer></body></html>"""
    with open(path, 'w') as f: f.write(html)
    print(f"  Saved → {path}")
