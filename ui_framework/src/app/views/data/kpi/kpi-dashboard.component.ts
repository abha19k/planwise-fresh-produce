// src/app/views/data/kpi/kpi-dashboard.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

import {
  CardComponent,
  CardHeaderComponent,
  CardBodyComponent,
  ColComponent,
  RowComponent,
} from '@coreui/angular';

type DimensionType = 'Location' | 'Product' | 'Channel';

interface KpiRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  Period: string;
  LagConfig: string;
  WMAPE: number | null;
  WAPE: number | null;
  MAE: number | null;
  RMSE: number | null;
  MAPE: number | null;
  sMAPE: number | null;
  Bias: number | null;
}

interface BarDatum {
  label: string;
  value: number;     // numeric metric value (e.g. accuracy %, MAPE, RMSE, Bias)
  heightPx: number;  // bar height in pixels
}

@Component({
  selector: 'app-kpi-dashboard',
  standalone: true,
  templateUrl: './kpi-dashboard.component.html',
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    ColComponent,
    RowComponent,
  ],
})
export class KpiDashboardComponent implements OnInit {
  // --- API base ---
  private readonly apiBase = 'http://localhost:8000';

  // --- Raw data from backend ---
  allRows: KpiRow[] = [];

  // --- Filters / controls ---
  dimensions: DimensionType[] = ['Location', 'Product', 'Channel'];
  dimension: DimensionType = 'Location';

  periods: string[] = [];
  selectedPeriod: string | null = null;

  lagConfigs: string[] = [];
  selectedLag: string | null = null;

  // --- Bar data for each metric ---
  accuracyBars: BarDatum[] = [];
  mapeBars: BarDatum[] = [];
  rmseBars: BarDatum[] = [];
  biasBars: BarDatum[] = [];

  // --- Dynamic ticks for Y axes ---
  accuracyTicks: number[] = [];
  mapeTicks: number[] = [];
  rmseTicks: number[] = [];
  biasTicks: number[] = [];

  // --- UI state ---
  loading = false;
  errorMessage = '';

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetchKpiRows();
  }

  // ======================================================
  // Data load
  // ======================================================
  private fetchKpiRows(): void {
    this.loading = true;
    this.errorMessage = '';

    const url = `${this.apiBase}/api/kpi/monthly-bycombo`;

    this.http.get<KpiRow[]>(url).subscribe({
      next: (rows) => {
        this.loading = false;
        this.allRows = (rows || []).map((r) => ({
          ...r,
          // Ensure numeric types (backend might send strings)
          WMAPE: r.WMAPE != null ? Number(r.WMAPE) : null,
          WAPE: r.WAPE != null ? Number(r.WAPE) : null,
          MAE: r.MAE != null ? Number(r.MAE) : null,
          RMSE: r.RMSE != null ? Number(r.RMSE) : null,
          MAPE: r.MAPE != null ? Number(r.MAPE) : null,
          sMAPE: r.sMAPE != null ? Number(r.sMAPE) : null,
          Bias: r.Bias != null ? Number(r.Bias) : null,
        }));

        this.initFilterOptions();
        this.rebuildBars();
      },
      error: (err) => {
        this.loading = false;
        console.error('Error loading KPI rows', err);
        this.errorMessage = 'Failed to load KPI data from server.';
      },
    });
  }

  private initFilterOptions(): void {
    // Distinct Periods
    const periodSet = new Set<string>();
    const lagSet = new Set<string>();

    for (const r of this.allRows) {
      if (r.Period) periodSet.add(r.Period);
      if (r.LagConfig) lagSet.add(r.LagConfig);
    }

    this.periods = Array.from(periodSet).sort();
    this.lagConfigs = Array.from(lagSet).sort();

    // Default selections
    if (!this.selectedPeriod && this.periods.length > 0) {
      this.selectedPeriod = this.periods[0];
    }
    if (!this.selectedLag && this.lagConfigs.length > 0) {
      this.selectedLag = this.lagConfigs[0];
    }
  }

  // ======================================================
  // Event handlers from HTML
  // ======================================================
  onDimensionChange(dim: DimensionType): void {
    this.dimension = dim;
    this.rebuildBars();
  }

  onPeriodChange(p: string): void {
    this.selectedPeriod = p;
    this.rebuildBars();
  }

  onLagChange(l: string): void {
    this.selectedLag = l;
    this.rebuildBars();
  }

  // ======================================================
  // Core bar + axis logic
  // ======================================================

  private getFilteredRows(): KpiRow[] {
    return this.allRows.filter((r) => {
      const periodOk = !this.selectedPeriod || r.Period === this.selectedPeriod;
      const lagOk = !this.selectedLag || r.LagConfig === this.selectedLag;
      return periodOk && lagOk;
    });
  }

  private getKeyForDimension(r: KpiRow): string {
    switch (this.dimension) {
      case 'Location':
        return r.LocationID || '(no location)';
      case 'Product':
        return r.ProductID || '(no product)';
      case 'Channel':
        return r.ChannelID || '(no channel)';
      default:
        return '';
    }
  }

  private rebuildBars(): void {
    const rows = this.getFilteredRows();

    this.accuracyBars = [];
    this.mapeBars = [];
    this.rmseBars = [];
    this.biasBars = [];

    this.accuracyTicks = [];
    this.mapeTicks = [];
    this.rmseTicks = [];
    this.biasTicks = [];

    if (!rows.length) {
      return;
    }

    // Group rows by selected dimension (Location / Product / Channel)
    const groups = new Map<
      string,
      { label: string; wmapes: number[]; mapes: number[]; rmses: number[]; biases: number[] }
    >();

    for (const r of rows) {
      const key = this.getKeyForDimension(r);
      const label = key;

      let g = groups.get(key);
      if (!g) {
        g = { label, wmapes: [], mapes: [], rmses: [], biases: [] };
        groups.set(key, g);
      }

      if (r.WMAPE != null && isFinite(r.WMAPE)) g.wmapes.push(r.WMAPE);
      if (r.MAPE != null && isFinite(r.MAPE)) g.mapes.push(r.MAPE);
      if (r.RMSE != null && isFinite(r.RMSE)) g.rmses.push(r.RMSE);
      if (r.Bias != null && isFinite(r.Bias)) g.biases.push(r.Bias);
    }

    const accBars: BarDatum[] = [];
    const mapeBars: BarDatum[] = [];
    const rmseBars: BarDatum[] = [];
    const biasBars: BarDatum[] = [];

    let maxAcc = 0;
    let maxMape = 0;
    let maxRmse = 0;
    let maxBiasAbs = 0;

    for (const [, g] of groups) {
      const avgWmape = this.mean(g.wmapes);
      const avgMape = this.mean(g.mapes);
      const avgRmse = this.mean(g.rmses);
      const avgBias = this.mean(g.biases);

      // If no data for a metric in this group, skip it for that metric
      if (avgWmape != null) {
        const accuracyPct = (1 - avgWmape) * 100;
        if (isFinite(accuracyPct)) {
          accBars.push({ label: g.label, value: accuracyPct, heightPx: 0 });
          maxAcc = Math.max(maxAcc, accuracyPct);
        }
      }

      if (avgMape != null && isFinite(avgMape)) {
        // avgMape is assumed already in %
        mapeBars.push({ label: g.label, value: avgMape, heightPx: 0 });
        maxMape = Math.max(maxMape, avgMape);
      }

      if (avgRmse != null && isFinite(avgRmse)) {
        rmseBars.push({ label: g.label, value: avgRmse, heightPx: 0 });
        maxRmse = Math.max(maxRmse, avgRmse);
      }

      if (avgBias != null && isFinite(avgBias)) {
        biasBars.push({ label: g.label, value: avgBias, heightPx: 0 });
        maxBiasAbs = Math.max(maxBiasAbs, Math.abs(avgBias));
      }
    }

    const MAX_BAR_HEIGHT = 170; // px of usable bar height

    // Scale bars based on max
    this.accuracyBars = accBars.map((b) => ({
      ...b,
      heightPx: maxAcc > 0 ? (b.value / maxAcc) * MAX_BAR_HEIGHT : 0,
    }));

    this.mapeBars = mapeBars.map((b) => ({
      ...b,
      heightPx: maxMape > 0 ? (b.value / maxMape) * MAX_BAR_HEIGHT : 0,
    }));

    this.rmseBars = rmseBars.map((b) => ({
      ...b,
      heightPx: maxRmse > 0 ? (b.value / maxRmse) * MAX_BAR_HEIGHT : 0,
    }));

    this.biasBars = biasBars.map((b) => ({
      ...b,
      heightPx: maxBiasAbs > 0 ? (Math.abs(b.value) / maxBiasAbs) * MAX_BAR_HEIGHT : 0,
    }));

    // Now build dynamic ticks for Y axes
    this.buildAllTicks(maxAcc, maxMape, maxRmse, maxBiasAbs);
  }

  // ======================================================
  // Tick helpers
  // ======================================================

  private buildAllTicks(
    maxAcc: number,
    maxMape: number,
    maxRmse: number,
    maxBiasAbs: number
  ): void {
    // Accuracy is a percent: keep fixed 0..100 (but still generated dynamically)
    this.accuracyTicks = this.buildTicks(0, 100, 5);

    // MAPE: 0..(rounded-up max)
    const mapeMax = this.roundUpNice(maxMape, 5) || 10;
    this.mapeTicks = this.buildTicks(0, mapeMax, 5);

    // RMSE: 0..(rounded-up max)
    const rmseMax = this.roundUpNice(maxRmse, 50) || 50;
    this.rmseTicks = this.buildTicks(0, rmseMax, 5);

    // Bias: symmetrical around 0
    const biasMax = maxBiasAbs || 1;
    this.biasTicks = this.buildSymmetricTicks(biasMax, 4);
  }

  /**
   * Build ticks from min..max in N equal steps (ascending).
   * Example: min=0, max=100, steps=5 => [0, 20, 40, 60, 80, 100]
   */
  private buildTicks(min: number, max: number, steps: number): number[] {
    if (!isFinite(max) || max <= min) {
      return [min, max];
    }
    const result: number[] = [];
    const step = (max - min) / steps;
    for (let i = 0; i <= steps; i++) {
      result.push(min + i * step);
    }
    return result;
  }

  /**
   * Build symmetrical ticks for Bias axis, e.g. [-X, -X/2, 0, X/2, X].
   */
  private buildSymmetricTicks(maxAbs: number, stepsEachSide: number): number[] {
    const ticks: number[] = [];
    const step = maxAbs / stepsEachSide;
    for (let i = -stepsEachSide; i <= stepsEachSide; i++) {
      ticks.push(i * step);
    }
    return ticks;
  }

  /**
   * Round a max value to a "nice" upper bound (like 10, 20, 50, 100, etc.).
   */
  private roundUpNice(maxVal: number, baseStep: number): number {
    if (!isFinite(maxVal) || maxVal <= 0) {
      return baseStep;
    }
    // Simple approach: round up to nearest multiple of baseStep
    const mult = Math.ceil(maxVal / baseStep);
    return mult * baseStep;
  }

  // ======================================================
  // Small numeric helper
  // ======================================================
  private mean(arr: number[]): number | null {
    const valid = arr.filter((v) => v != null && isFinite(v));
    if (!valid.length) return null;
    const sum = valid.reduce((acc, v) => acc + v, 0);
    return sum / valid.length;
  }
}
