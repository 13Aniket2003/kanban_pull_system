import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('kanban_software_dataset.csv')
df['created_date']   = pd.to_datetime(df['created_date'])
df['started_date']   = pd.to_datetime(df['started_date'])
df['completed_date'] = pd.to_datetime(df['completed_date'])
df['flow_efficiency'] = ((df['lead_time_days'] - df['blocked_days']) / df['lead_time_days']) * 100

tier_order = ['S', 'M', 'L', 'XL']
tier_colors = {'S': '#27ae60', 'M': '#f39c12', 'L': '#e74c3c', 'XL': '#8e44ad'}

# ─── WIP VIOLATION ANALYSIS ───────────────────────────────────────────────────
violated     = df[df['wip_violations'] == 'Yes']
clean        = df[df['wip_violations'] == 'No']
viol_rate    = len(violated) / len(df) * 100
avg_ct_viol  = violated['cycle_time_days'].mean()
avg_ct_clean = clean['cycle_time_days'].mean()
ct_increase  = (avg_ct_viol - avg_ct_clean) / avg_ct_clean * 100

print("=== WIP VIOLATION ANALYSIS ===")
print(f"Total items              : {len(df)}")
print(f"Items WITH violations    : {len(violated)} ({viol_rate:.1f}%)")
print(f"Items WITHOUT violations : {len(clean)} ({100-viol_rate:.1f}%)")
print(f"Avg cycle time WITH violations    : {avg_ct_viol:.1f} days")
print(f"Avg cycle time WITHOUT violations : {avg_ct_clean:.1f} days")
print(f"Cycle time increase due to violations : {ct_increase:.1f}%")
print()

# ─── CYCLE TIME ANALYSIS BY SIZE TIER ────────────────────────────────────────
ct_stats = df.groupby('size_tier')['cycle_time_days'].agg(
    Count='count', Mean='mean', Median='median', Std='std', Min='min', Max='max'
).reindex(tier_order)

print("=== CYCLE TIME BY SIZE TIER (DAKAM - Work Item Homogeneity) ===")
print(ct_stats.round(2))
xl_mean = ct_stats.loc['XL', 'Mean']
s_mean  = ct_stats.loc['S',  'Mean']
print(f"XL items take {xl_mean/s_mean:.0f}x longer than S items")
cv = df['cycle_time_days'].std() / df['cycle_time_days'].mean() * 100
print(f"Coefficient of Variation across all items : {cv:.1f}%")
print()

# ─── WASTE TAXONOMY (Ohno's 7 Muda + DAKAM 8th) ──────────────────────────────
n = len(df)
muda = {
    '1. Overproduction'   : (df['size_tier'] == 'XL').sum(),
    '2. Waiting'          : (df['blocked_days'] > 0).sum(),
    '3. Transport'        : (df['lead_time_days'] - df['cycle_time_days'] > 2).sum(),
    '4. Over-processing'  : (df['context_switches'] > 2).sum(),
    '5. Inventory'        : (df['lead_time_days'] > 2 * df['cycle_time_days']).sum(),
    '6. Motion'           : ((df['rework_required'] == 'Yes') & (df['defects_found'] == 0)).sum(),
    '7. Defects'          : (df['defects_found'] > 0).sum(),
    '8. Ambiguity [DAKAM]': (df['ambiguity_waste'] == 'Yes').sum(),
}

print("=== WASTE TAXONOMY - Ohno's 7 Muda + DAKAM 8th Ambiguity Waste ===")
for cat, cnt in muda.items():
    print(f"  {cat:<25} : {cnt:>3} items  ({cnt/n*100:.1f}%)")
print(f"  Total blocked days (Waiting Muda) : {df['blocked_days'].sum()}")
print()

# ─── FLOW EFFICIENCY (Heijunka gap) ──────────────────────────────────────────
overall_fe = df['flow_efficiency'].mean()
spikes     = (df['heijunka_demand_spike'] == 'Yes').sum()
avg_ctx    = df['context_switches'].mean()

print("=== FLOW EFFICIENCY ANALYSIS (Heijunka / Flow Levelling) ===")
print(f"Overall Flow Efficiency            : {overall_fe:.1f}%")
print(f"Manufacturing benchmark            : ~85%+")
print(f"Software industry average          : ~15-25%")
print(f"Items with unlevelled demand spikes: {spikes} ({spikes/n*100:.1f}%)")
print(f"Average context switches per item  : {avg_ctx:.1f}")
fe_by_tier = df.groupby('size_tier')['flow_efficiency'].mean().reindex(tier_order)
print("Flow efficiency by tier:")
for tier, fe in fe_by_tier.items():
    print(f"  {tier}: {fe:.1f}%")
print()

# ─── DAKAM FIDELITY SCORES (Table 1 of paper) ────────────────────────────────
violation_rate = (df['wip_violations'] == 'Yes').mean() * 100
rework_rate    = (df['rework_required'] == 'Yes').mean() * 100
spike_rate     = (df['heijunka_demand_spike'] == 'Yes').mean() * 100
retro_created  = (df['retrospective_item_created'] == 'Yes').mean() * 100
retro_resolved = (df['improvement_item_resolved'] == 'Yes').mean() * 100

dakam_scores = {
    'Pull Signal Mechanism'   : round(100 - violation_rate * 0.65, 1),
    'WIP Enforcement Mode'    : round(100 - violation_rate, 1),
    'Work Item Homogeneity'   : round(max(0, 100 - cv * 0.55), 1),
    'Waste Taxonomy'          : round(max(0, 100 - rework_rate * 1.1), 1),
    'Flow Levelling'          : round(max(0, 100 - spike_rate * 1.3), 1),
    'Feedback Loop Structure' : round(retro_created, 1),
    'Improvement Governance'  : round(retro_resolved, 1),
}

print("=== DAKAM FIDELITY SCORES (Table 1) ===")
for dim, score in dakam_scores.items():
    rating = 'HIGH' if score >= 70 else ('MEDIUM' if score >= 45 else 'LOW')
    print(f"  {dim:<28} : {score:>5}/100  [{rating}]")
print(f"  Overall DAKAM Fidelity : {np.mean(list(dakam_scores.values())):.1f}/100")
print()

# ─── CROSS-DOMAIN COMPARISON (Table 2 of paper) ──────────────────────────────
print("=== CROSS-DOMAIN COMPARISON (Table 2) ===")
params = [
    ("WIP Signal Type",
     "Physical card - structural constraint",
     f"Column limit - cultural convention ({len(violated)} violations found)"),
    ("Cycle Time Predictability",
     "HIGH - homogeneous units, stable processing",
     f"LOW - CV={cv:.0f}%, range {df['cycle_time_days'].min()}-{df['cycle_time_days'].max()} days"),
    ("WIP Enforcement",
     "Impossible to exceed (no card = no production)",
     f"Advisory only - {violation_rate:.0f}% items violated limits"),
    ("Flow Levelling",
     "Heijunka scheduling - structurally present",
     f"None - {spikes} unlevelled demand spikes detected"),
    ("Waste Identification",
     "Value stream mapping of physical flow",
     f"CFD + retrospectives - {df['blocked_days'].sum()} total blocked days"),
    ("Improvement Governance",
     "Formal PDCA Kaizen with tracked outcomes",
     f"Retrospectives - {int(retro_resolved)}% items resolved"),
]
for param, mfg, sw in params:
    print(f"\n  {param}")
    print(f"    Manufacturing : {mfg}")
    print(f"    Software      : {sw}")
print()

# ─── PROPOSED SOLUTIONS (Section 6 of paper) ─────────────────────────────────
print("=== PROPOSED SOLUTIONS (Section 6) ===")
solutions = [
    ("1", "Automate WIP enforcement via tooling policy gates",
     f"{violation_rate:.0f}% violation rate - advisory model fails",
     "Block state transitions in Jira/Azure DevOps when WIP limit reached"),
    ("2", "Size classification before board implementation (S/M/L/XL)",
     f"XL items {xl_mean/s_mean:.0f}x longer than S - single WIP limit is invalid",
     "Tier-specific WIP limits per column"),
    ("3", "Capacity-based intake ceremonies (Heijunka equivalent)",
     f"{spikes} demand spikes - no levelling mechanism exists",
     "Accept items only up to measured team capacity per period"),
    ("4", "Mandatory improvement items per retrospective (PDCA)",
     f"{int(retro_resolved)}% improvement items resolved - governance gap",
     "One board item per retro with named owner and deadline"),
]
for num, title, evidence, action in solutions:
    print(f"\n  Solution #{num}: {title}")
    print(f"    Evidence : {evidence}")
    print(f"    Action   : {action}")
print()

# =============================================================================
# FIGURE 1 — WIP Enforcement: Manufacturing (0%) vs Software
# (Divergence #1 from paper Section 5, Sjoberg [6])
# =============================================================================
fig1, ax = plt.subplots(figsize=(7, 5))
categories = ['Manufacturing\nKanban', 'Software\nKanban']
clean_vals = [100, 100 - viol_rate]
viol_vals  = [0,   viol_rate]
ax.bar(categories, clean_vals, color=['#2c3e50', '#2980b9'], alpha=0.8,
       label='Clean (no violation)', edgecolor='black', width=0.5)
ax.bar(categories, viol_vals, bottom=clean_vals, color=['#bdc3c7', '#e74c3c'],
       alpha=0.9, label='WIP Violated', edgecolor='black', width=0.5)
ax.text(0, 50, '0%\nviolations\n(card count\nprevents it)',
        ha='center', color='white', fontsize=9, fontweight='bold')
ax.text(1, 100 - viol_rate / 2 + 5, f'{viol_rate:.0f}%\nviolated',
        ha='center', color='white', fontsize=10, fontweight='bold')
ax.set_ylabel('% of Work Items', fontsize=11)
ax.set_ylim(0, 130)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_wip_enforcement.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig1_wip_enforcement.png")

# =============================================================================
# FIGURE 2 — Cycle Time by Size Tier Box Plot
# (Divergence #2 from paper Section 5 — Work Item Homogeneity)
# =============================================================================
fig2, ax = plt.subplots(figsize=(7, 5))
data_by_tier = [df[df['size_tier'] == t]['cycle_time_days'].values for t in tier_order]
bp = ax.boxplot(data_by_tier, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
for patch, tier in zip(bp['boxes'], tier_order):
    patch.set_facecolor(tier_colors[tier])
    patch.set_alpha(0.75)
ax.set_xticklabels(
    [f'{t}\n(n={df[df["size_tier"]==t].shape[0]})' for t in tier_order],
    fontsize=10
)
ax.set_ylabel('Cycle Time (days)', fontsize=11)
ax.set_xlabel('Size Tier', fontsize=11)
ax.grid(axis='y', alpha=0.3)
patches = [mpatches.Patch(color=tier_colors[t], label=t, alpha=0.75) for t in tier_order]
ax.legend(handles=patches, fontsize=9)
plt.tight_layout()
plt.savefig('fig2_cycle_time_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig2_cycle_time_by_tier.png")

# =============================================================================
# FIGURE 3 — Ohno's 7 Muda + DAKAM 8th Ambiguity Waste
# (DAKAM Dimension 4: Waste Taxonomy, paper Section 5)
# =============================================================================
fig3, ax = plt.subplots(figsize=(8, 5))
muda_labels = list(muda.keys())
muda_pcts   = [v / n * 100 for v in muda.values()]
colors_muda = ['#2c3e50'] * 7 + ['#c0392b']
bars = ax.barh(muda_labels, muda_pcts, color=colors_muda, alpha=0.85, edgecolor='black')
for bar, pct in zip(bars, muda_pcts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pct:.0f}%', va='center', fontsize=9)
ax.set_xlabel('% of Work Items Affected', fontsize=11)
ax.set_xlim(0, 85)
ax.tick_params(axis='y', labelsize=9)
ax.grid(axis='x', alpha=0.3)
legend_handles = [
    mpatches.Patch(color='#2c3e50', label="Ohno's 7 Muda"),
    mpatches.Patch(color='#c0392b', label='DAKAM 8th: Ambiguity Waste'),
]
ax.legend(handles=legend_handles, fontsize=9)
plt.tight_layout()
plt.savefig('fig3_waste_taxonomy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig3_waste_taxonomy.png")

# =============================================================================
# FIGURE 4 — DAKAM Fidelity Scores (all 7 dimensions from Table 1)
# (paper Section 5, Table 1)
# =============================================================================
fig4, ax = plt.subplots(figsize=(8, 5))
dims   = list(dakam_scores.keys())
scores = list(dakam_scores.values())
bar_colors = [
    '#27ae60' if s >= 70 else '#f39c12' if s >= 45 else '#e74c3c'
    for s in scores
]
bars = ax.barh(dims, scores, color=bar_colors, alpha=0.85, edgecolor='black')
ax.axvline(70, color='green',  linestyle='--', linewidth=1.5, label='HIGH threshold (70)')
ax.axvline(45, color='orange', linestyle='--', linewidth=1.5, label='MEDIUM threshold (45)')
for bar, val in zip(bars, scores):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}', va='center', fontsize=9)
ax.set_xlabel('Fidelity Score (0–100)', fontsize=11)
ax.set_xlim(0, 125)
ax.tick_params(axis='y', labelsize=9)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
legend_handles = [
    mpatches.Patch(color='#27ae60', label='HIGH (≥70)'),
    mpatches.Patch(color='#f39c12', label='MEDIUM (45–69)'),
    mpatches.Patch(color='#e74c3c', label='LOW (<45)'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('fig4_dakam_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig4_dakam_scores.png")

# =============================================================================
# FIGURE 5 — Flow Levelling Gap (Heijunka vs Software demand spikes)
# (Divergence #3 from paper Section 5, Monden [3], Bullwhip effect Lee [12])
# =============================================================================
fig5, ax = plt.subplots(figsize=(8, 5))
df_weekly   = df.copy()
df_weekly['week'] = df_weekly['completed_date'].dt.isocalendar().week
weekly_total  = df_weekly.groupby('week').size()
weekly_spikes = df_weekly.groupby('week').apply(
    lambda x: (x['heijunka_demand_spike'] == 'Yes').sum()
)
ax.bar(weekly_total.index, weekly_total.values, color='#2980b9',
       alpha=0.75, label='Total items completed per week', edgecolor='black', width=0.6)
ax.bar(weekly_spikes.index, weekly_spikes.values, color='#8e44ad',
       alpha=0.9, label='Items with demand spikes (no Heijunka)', edgecolor='black', width=0.6)
ax.axhline(weekly_total.mean(), color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Average = {weekly_total.mean():.1f} (ideal Heijunka level)')
ax.set_xlabel('Week Number', fontsize=11)
ax.set_ylabel('Number of Items', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_flow_levelling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig5_flow_levelling.png")

# =============================================================================
# FIGURE 6 — Cumulative Flow Diagram (CFD)
# (Vacanti [11] flow metric, paper Section 7 Best Practices)
# =============================================================================
fig6, ax = plt.subplots(figsize=(8, 5))
df_sorted  = df.sort_values('completed_date').reset_index(drop=True)
cumulative = np.arange(1, len(df_sorted) + 1)
ax.fill_between(df_sorted['completed_date'], cumulative,
                alpha=0.35, color='#2980b9', label='Cumulative items Done')
ax.plot(df_sorted['completed_date'], cumulative, color='#2980b9', linewidth=2)
spike_mask    = df_sorted['heijunka_demand_spike'] == 'Yes'
spike_indices = cumulative[spike_mask]
ax.scatter(df_sorted.loc[spike_mask, 'completed_date'], spike_indices,
           color='#e74c3c', zorder=5, s=30, label='Demand spike point', alpha=0.8)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Cumulative Items Completed', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.tick_params(axis='x', rotation=30, labelsize=8)
plt.tight_layout()
plt.savefig('fig6_cumulative_flow_diagram.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig6_cumulative_flow_diagram.png")

print("\nAll figures saved individually.")
