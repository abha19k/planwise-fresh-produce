import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
  inject
} from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import Chart from 'chart.js/auto';

import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  ColComponent,
  RowComponent
} from '@coreui/angular';

import { IconDirective } from '@coreui/icons-angular';

import {
  Scenario,
  ScenarioService
} from '../../services/scenario.service';

interface DashboardSummary {
  forecastAccuracy: number | null;
  forecastBias: number | null;
  forecastElements: number | null;
  needsAttention: number | null;
}

interface ForecastElementsResponse {
  count: number;
  rows: any[];
}


interface DashboardKpiResponse {
  scenario_id: number;
  period: string;
  level: string;
  lag: number;
  n: number;
  wape: number | null;
  accuracy: number | null;
  bias_pct: number | null;
}

interface DashboardHistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  Period?: string;
  Qty: number;
}

interface DashboardForecastRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  Period?: string;
  ForecastQty: number;
}

type DashboardHistoryResponse = DashboardHistoryRow[];


interface DashboardForecastResponse {
  rows?: DashboardForecastRow[];
  data?: DashboardForecastRow[];
}

interface DashboardBacktestResponse {
  ok: boolean;

  scenario_id: number;
  level: string;
  period: string;
  variant: string;

  product_id?: string;
  channel_id?: string;
  location_id?: string;

  accuracy?: number | null;
  wape?: number | null;
  bias_pct?: number | null;

  n?: number;
  folds?: number;

  run_at?: string;
  reason?: string;
}


@Component({
  selector: 'app-dashboard',
  standalone: true,
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  imports: [
    CommonModule,
    RowComponent,
    ColComponent,
    CardComponent,
    CardBodyComponent,
    ButtonDirective,
    IconDirective
  ]
})

export class DashboardComponent
  implements OnInit, AfterViewInit, OnDestroy {

  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);

  readonly scenarioService = inject(ScenarioService);

  private readonly apiBase = 'http://127.0.0.1:8000';

  @ViewChild('forecastChart')
  forecastChartRef?: ElementRef<HTMLCanvasElement>;

  private forecastChart?: Chart;
  private viewReady = false;


  readonly dashboardForecastElement = {
    ProductID: 'Blueberry_Ventura',
    ChannelID: 'JUMBO',
    LocationID: 'NL-NH',
    Period: 'Weekly',
    Variant: 'feat',
    Lag: 1
  };

  chartLoading = false;
  chartError = '';

  loading = false;
  errorMessage = '';

  summary: DashboardSummary = {
    forecastAccuracy: null,
    forecastBias: null,
    forecastElements: null,
    needsAttention: null
  };

  ngOnInit(): void {
    this.loadScenarios();
  }

  ngAfterViewInit(): void {
    this.viewReady = true;
  }
  
  ngOnDestroy(): void {
    this.forecastChart?.destroy();
  }

  // ---------------------------------------------------------
  // SCENARIO
  // ---------------------------------------------------------

  private loadScenarios(): void {
    this.loading = true;
    this.errorMessage = '';

    this.http
      .get<Scenario[]>(`${this.apiBase}/api/scenarios`)
      .subscribe({
        next: (scenarios) => {
          this.scenarioService.setScenarios(scenarios ?? []);

          this.loadDashboardSummary();
          this.loadForecastChart();
        },

        error: (err) => {
          console.error('Dashboard scenario load failed', err);

          this.loading = false;

          this.errorMessage =
            err?.error?.detail ||
            'Unable to load dashboard information.';
        }
      });
  }

  // ---------------------------------------------------------
  // DASHBOARD SUMMARY
  // ---------------------------------------------------------

  private loadDashboardSummary(): void {

    const scenarioId =
      this.scenarioService.selectedScenarioId();
  
    const key =
      this.dashboardForecastElement;
  
    this.loading = true;
    this.errorMessage = '';
  
    // ---------------------------------------------------------
    // 1. Load total forecast-element count
    // ---------------------------------------------------------
  
    this.http
      .get<ForecastElementsResponse>(
        `${this.apiBase}/api/forecastelements`,
        {
          params: {
            limit: 1
          }
        }
      )
      .subscribe({
  
        next: (elementsRes) => {
  
          this.summary = {
            ...this.summary,
            forecastElements: elementsRes.count ?? 0
          };
  
          // -----------------------------------------------------
          // 2. Load rolling-backtest performance
          // -----------------------------------------------------
  
          this.http
            .get<DashboardBacktestResponse>(
              `${this.apiBase}/api/forecast/backtest-latest`,
              {
                params: {
                  db_schema: 'planwise_fresh_produce',
                  scenario_id: scenarioId,
                  level: '111',
  
                  period: key.Period,
                  variant: key.Variant,
  
                  productid: key.ProductID,
                  channelid: key.ChannelID,
                  locationid: key.LocationID
                }
              }
            )
            .subscribe({
  
              next: (backtestRes) => {
  
                if (!backtestRes.ok) {
  
                  console.warn(
                    'Dashboard backtest unavailable',
                    backtestRes
                  );
  
                  this.summary = {
                    ...this.summary,
                    forecastAccuracy: null,
                    forecastBias: null
                  };
  
                  this.loading = false;
  
                  return;
                }
  
                this.summary = {
                  ...this.summary,

                  forecastAccuracy: backtestRes.accuracy ?? null,
                  forecastBias: backtestRes.bias_pct ?? null
  
                };
  
                this.loading = false;
  
                console.log(
                  'Dashboard backtest summary',
                  {
                    accuracy: backtestRes.accuracy,
                    wape: backtestRes.wape,
                    bias: backtestRes.bias_pct,
                    observations: backtestRes.n,
                    folds: backtestRes.folds
                  }
                );
              },
  
              error: (err) => {
  
                console.error(
                  'Dashboard backtest load failed',
                  err
                );
  
                this.summary = {
                  ...this.summary,
                  forecastAccuracy: null,
                  forecastBias: null
                };
  
                this.loading = false;
  
                this.errorMessage =
                  err?.error?.detail ||
                  'Unable to load forecast performance.';
              }
            });
        },
  
        error: (err) => {
  
          console.error(
            'Dashboard forecast-element load failed',
            err
          );
  
          this.loading = false;
  
          this.errorMessage =
            err?.error?.detail ||
            'Unable to load dashboard information.';
        }
      });
  }


  // ---------------------------------------------------------
  // DISPLAY HELPERS
  // ---------------------------------------------------------

  get activeScenarioName(): string {
    return (
      this.scenarioService.selectedScenario()?.name ||
      'Base'
    );
  }

  get activeScenarioStatus(): string {
    return (
      this.scenarioService.selectedScenario()?.status ||
      'Active'
    );
  }

  get greeting(): string {
    const hour = new Date().getHours();

    if (hour < 12) {
      return 'Good morning';
    }

    if (hour < 18) {
      return 'Good afternoon';
    }

    return 'Good evening';
  }

  formatPercent(value: number | null): string {
    if (value === null || value === undefined) {
      return '—';
    }

    return `${value.toFixed(1)}%`;
  }

  formatNumber(value: number | null): string {
    if (value === null || value === undefined) {
      return '—';
    }

    return value.toLocaleString();
  }

  get accuracyStatus(): 'good' | 'warning' | 'danger' | 'neutral' {
    const value = this.summary.forecastAccuracy;
  
    if (value === null) return 'neutral';
    if (value >= 80) return 'good';
    if (value >= 65) return 'warning';
  
    return 'danger';
  }
  
  get accuracyMessage(): string {
    const value = this.summary.forecastAccuracy;
  
    if (value === null) {
      return 'Forecast accuracy not available';
    }
  
    if (value >= 80) {
      return `${value.toFixed(1)}% — forecast performing well`;
    }
  
    if (value >= 65) {
      return `${value.toFixed(1)}% — accuracy could be improved`;
    }
  
    return `${value.toFixed(1)}% — forecast needs attention`;
  }
  
  get biasStatus(): 'good' | 'warning' | 'danger' | 'neutral' {
    const value = this.summary.forecastBias;
  
    if (value === null) return 'neutral';
  
    const absBias = Math.abs(value);
  
    if (absBias <= 5) return 'good';
    if (absBias <= 15) return 'warning';
  
    return 'danger';
  }

  get biasTitle(): string {
    const value = this.summary.forecastBias;
  
    if (value === null) {
      return 'Forecast bias';
    }
  
    const absBias = Math.abs(value);
  
    if (absBias <= 5) {
      return 'Forecast well balanced';
    }
  
    if (absBias <= 15) {
      return value > 0
        ? 'Moderate over-forecasting'
        : 'Moderate under-forecasting';
    }
  
    return value > 0
      ? 'High over-forecasting'
      : 'High under-forecasting';
  }
  

  
  get biasMessage(): string {
    const value = this.summary.forecastBias;
  
    if (value === null) {
      return 'Forecast bias not available';
    }
  
    if (Math.abs(value) <= 5) {
      return `${value.toFixed(1)}% — bias is within target range`;
    }
  
    return `${value > 0 ? '+' : ''}${value.toFixed(1)}% forecast bias`;
  }
  
  get coverageMessage(): string {
    const value = this.summary.forecastElements;
  
    if (value === null) {
      return 'Forecast coverage not available';
    }
  
    return `${value.toLocaleString()} forecast elements available`;
  }

  // ---------------------------------------------------------
  // FORECAST PERFORMANCE CHART
  // ---------------------------------------------------------

  private loadForecastChart(): void {

    const key = this.dashboardForecastElement;
    const scenarioId =
      this.scenarioService.selectedScenarioId();

    this.chartLoading = true;
    this.chartError = '';

    const historyBody = {
      keys: [
        {
          ProductID: key.ProductID,
          ChannelID: key.ChannelID,
          LocationID: key.LocationID
        }
      ]
    };

    this.http
      .post<DashboardHistoryResponse>(
        `${this.apiBase}/api/history/weekly-by-keys`,
        historyBody,
        {
          params: {
            db_schema: 'planwise_fresh_produce',
            limit_per_key: 5000
          }
        }
      )
      .subscribe({

        next: (historyRes) => {

          
          const historyRows =
            Array.isArray(historyRes)
              ? historyRes
              : [];  

          console.log(
              'Dashboard history response',
              historyRes
          );
            
          console.log(
              'Dashboard history rows',
              historyRows
          );

          this.http
            .get<DashboardForecastResponse>(
              `${this.apiBase}/api/forecast`,
              {
                params: {
                  db_schema: 'planwise_fresh_produce',
                  scenario_id: scenarioId,
                  productid: key.ProductID,
                  channelid: key.ChannelID,
                  locationid: key.LocationID,
                  period: key.Period,
                  variant: key.Variant,
                  limit: 5000,
                  offset: 0
                }
              }
            )
            .subscribe({

              next: (forecastRes) => {

                const forecastRows =
                  forecastRes.rows ??
                  forecastRes.data ??
                  [];

                console.log(
                    'Dashboard forecast response',
                    forecastRes
                );
                  
                console.log(
                    'Dashboard forecast rows',
                    forecastRows
                ); 
                
                this.chartLoading = false;

                this.buildForecastChart(
                  historyRows,
                  forecastRows
                );

              
              },

              error: (err) => {

                console.error(
                  'Dashboard forecast chart load failed',
                  err
                );

                this.chartLoading = false;
                this.chartError =
                  'Unable to load forecast data.';
              }
            });
        },

        error: (err) => {

          console.error(
            'Dashboard history chart load failed',
            err
          );

          this.chartLoading = false;
          this.chartError =
            'Unable to load historical data.';
        }
      });
  }

  private buildForecastChart(
    historyRows: DashboardHistoryRow[],
    forecastRows: DashboardForecastRow[]
  ): void {

    
    this.chartError = '';
  
    // ---------------------------------------------------------
    // 1. Prepare historical actuals
    // ---------------------------------------------------------
  
    const actualPoints = historyRows
      .map(row => ({
        date: this.toDateKey(row.StartDate),
        value: Number(row.Qty ?? 0)
      }))
      .filter(point => !!point.date)
      .sort((a, b) => a.date.localeCompare(b.date));
  
    // Last 12 historical weeks
    const recentActuals = actualPoints.slice(-12);
  
  
    // ---------------------------------------------------------
    // 2. Prepare future forecast
    // ---------------------------------------------------------
  
    const forecastPoints = forecastRows
      .map(row => ({
        date: this.toDateKey(row.StartDate),
        value: Number(row.ForecastQty ?? 0)
      }))
      .filter(point => !!point.date)
      .sort((a, b) => a.date.localeCompare(b.date));
  
  
    if (!recentActuals.length && !forecastPoints.length) {
      this.chartError =
        'No actual or forecast data available.';
  
      this.forecastChart?.destroy();
      this.forecastChart = undefined;
  
      return;
    }
  
  
    // ---------------------------------------------------------
    // 3. Build one continuous date axis
    // ---------------------------------------------------------
  
    const allDates = [
      ...recentActuals.map(point => point.date),
      ...forecastPoints.map(point => point.date)
    ];
  
    const uniqueDates = [
      ...new Set(allDates)
    ].sort();
  
  
    // ---------------------------------------------------------
    // 4. Create maps for Chart.js
    // ---------------------------------------------------------
  
    const actualMap = new Map(
      recentActuals.map(point => [
        point.date,
        point.value
      ])
    );
  
    const forecastMap = new Map(
      forecastPoints.map(point => [
        point.date,
        point.value
      ])
    );
  
  
    // ---------------------------------------------------------
    // 5. Build datasets
    // ---------------------------------------------------------
  
    const actualData =
      uniqueDates.map(date =>
        actualMap.has(date)
          ? actualMap.get(date)!
          : null
      );
  
    const forecastData =
      uniqueDates.map(date =>
        forecastMap.has(date)
          ? forecastMap.get(date)!
          : null
      );
  
  
    console.log(
      'Dashboard actual points',
      recentActuals
    );
  
    console.log(
      'Dashboard future forecast points',
      forecastPoints
    );
  
  
    // ---------------------------------------------------------
    // 6. Render
    // ---------------------------------------------------------
  
    setTimeout(() => {
  
      const canvas =
        this.forecastChartRef?.nativeElement;
  
      if (!canvas) {
        this.chartError =
          'Forecast chart could not be rendered.';
        return;
      }
  
      this.forecastChart?.destroy();
  
  
      this.forecastChart = new Chart(
        canvas,
        {
          type: 'line',
  
          data: {
  
            labels: uniqueDates.map(
              date =>
                this.formatChartDate(date)
            ),
  
            datasets: [
  
              {
                label: 'Actual',
  
                data: actualData,
  
                borderWidth: 2.5,
  
                pointRadius: 2,
  
                pointHoverRadius: 5,
  
                tension: 0.3,
  
                spanGaps: false
              },
  
              {
                label: 'Forecast',
  
                data: forecastData,
  
                borderWidth: 2.5,
  
                pointRadius: 2,
  
                pointHoverRadius: 5,
  
                tension: 0.3,
  
                borderDash: [6, 4],
  
                spanGaps: false
              }
            ]
          },
  
          options: {
  
            responsive: true,
  
            maintainAspectRatio: false,
  
            interaction: {
              mode: 'index',
              intersect: false
            },
  
            plugins: {
  
              legend: {
                position: 'top',
                align: 'end'
              },
  
              tooltip: {
  
                callbacks: {
  
                  label: (context) => {
  
                    if (context.raw === null) {
                      return '';
                    }
  
                    const value =
                      Number(context.raw ?? 0);
  
                    return (
                      `${context.dataset.label}: ` +
                      value.toLocaleString(
                        undefined,
                        {
                          maximumFractionDigits: 0
                        }
                      )
                    );
                  }
                }
              }
            },
  
            scales: {
  
              x: {
  
                grid: {
                  display: false
                },
  
                ticks: {
                  maxRotation: 0,
                  autoSkip: true,
                  maxTicksLimit: 10
                }
              },
  
              y: {
  
                beginAtZero: false,
  
                ticks: {
  
                  callback: (value) =>
                    Number(value)
                      .toLocaleString()
                }
              }
            }
          }
        }
      );
  
    }, 0);
  }



  private toDateKey(
    value: string | null | undefined
  ): string {
  
    if (!value) return '';
  
    return value.slice(0, 10);
  }
  
  
  private shiftWeeklyDate(
    dateKey: string,
    weeks: number
  ): string {
  
    const date =
      new Date(`${dateKey}T00:00:00`);
  
    date.setDate(
      date.getDate() + weeks * 7
    );
  
    return [
      date.getFullYear(),
      String(date.getMonth() + 1)
        .padStart(2, '0'),
      String(date.getDate())
        .padStart(2, '0')
    ].join('-');
  }
  
  
  private formatChartDate(
    dateKey: string
  ): string {
  
    const date =
      new Date(`${dateKey}T00:00:00`);
  
    return date.toLocaleDateString(
      undefined,
      {
        day: '2-digit',
        month: 'short'
      }
    );
  }

  // ---------------------------------------------------------
  // NAVIGATION
  // ---------------------------------------------------------

  goToForecastTuning(): void {
    this.router.navigate(['/process/forecast-tuning']);
  }

  goToKpis(): void {
    this.router.navigate(['/data/kpi']);
  }

  goToScenarios(): void {
    this.router.navigate(['/process/scenario-manager']);
  }

  goToData(): void {
    this.router.navigate(['/data/history']);
  }

}