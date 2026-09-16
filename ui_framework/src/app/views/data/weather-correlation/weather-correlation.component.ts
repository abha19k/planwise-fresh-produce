// src/app/views/data/weather-correlation/weather-correlation.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';

import {
  CardComponent,
  CardHeaderComponent,
  CardBodyComponent,
  CardFooterComponent,
  ColComponent,
  RowComponent,
  TableDirective,
} from '@coreui/angular';

type PeriodType = 'daily' | 'weekly' | 'monthly';
type AnalysisTab =
  | 'correlation'
  | 'forecastImpact'
  | 'contribution'
  | 'featureImportance';

interface Product {
  ProductID: string;
  ProductDescr?: string | null;
}

interface Channel {
  ChannelID: string;
  ChannelDescr?: string | null;
}

interface Location {
  LocationID: string;
  LocationDescr?: string | null;
  Geography?: string | null;
}

interface WeatherSalesPoint {
  date: string;
  weatherMetric: number;
  quantity: number;
}

interface ScatterPoint {
  x: number;        // pixel X
  y: number;        // pixel Y
  date: string;
  weatherMetric: number;
  quantity: number;
}

interface CorrelationResponse {
  TempAvg: number | null;
  TempMin: number | null;
  TempMax: number | null;
  RainMm: number | null;
  SnowCm: number | null;
  WindMax: number | null;
  SunHours: number | null;
  [key: string]: number | null;
}

interface ForecastImpactResponse {
  scenario_id: number;
  level: string;
  ok: boolean;
  period: string;

  baseline: {
    accuracy: number | null;
    wape: number | null;
    bias_pct: number | null;
    folds: number;
    n: number;
  };

  feature: {
    accuracy: number | null;
    wape: number | null;
    bias_pct: number | null;
    folds: number;
    n: number;
  };

  accuracy_improvement: number | null;
  wape_reduction: number | null;
  external_factors_improved: boolean;
}

interface FactorMetrics {
  accuracy: number | null;
  wape: number | null;
  bias_pct: number | null;
  folds: number;
  n: number;
}

interface FactorContributionResponse {
  scenario_id: number;
  level: string;
  ok: boolean;
  period: string;

  forecast_key: {
    ProductID: string;
    ChannelID: string;
    LocationID: string;
  };

  baseline: FactorMetrics;
  weather: FactorMetrics;
  promotions: FactorMetrics;
  all: FactorMetrics;

  contribution: {
    weather_accuracy_pp: number | null;
    promotions_accuracy_pp: number | null;
    all_external_factors_accuracy_pp: number | null;
  };

  cached: boolean;
}

interface FeatureImportanceItem {
  feature: string;
  group: string;
  gain: number;
  importance_pct: number;
}

interface FeatureImportanceGroup {
  group: string;
  importance_pct: number;
}

interface FeatureImportanceResponse {
  scenario_id: number;
  level: string;
  ok: boolean;
  period: string;

  forecast_key: {
    ProductID: string;
    ChannelID: string;
    LocationID: string;
  };

  training_rows: number;
  feature_count: number;

  features: FeatureImportanceItem[];
  groups: FeatureImportanceGroup[];

  reason?: string;
}


@Component({
  selector: 'app-weather-correlation',
  standalone: true,
  templateUrl: './weather-correlation.component.html',
  imports: [
    CommonModule,
    FormsModule,
    
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    CardFooterComponent,
    RowComponent,
    ColComponent,
    TableDirective,
  ],
})
export class WeatherCorrelationComponent implements OnInit {
  private readonly apiBase = 'http://localhost:8000';

  // Dropdown data
  products: Product[] = [];
  channels: Channel[] = [];
  locations: Location[] = [];

  // Selections
  selectedProductId = '';
  selectedChannelId = '';
  selectedLocationId = '';

  periods: PeriodType[] = ['daily', 'weekly', 'monthly'];
  selectedPeriod: PeriodType | '' = '';

  metricOptions = [
    { key: 'TempAvg', label: 'Average Temperature' },
    { key: 'TempMin', label: 'Minimum Temperature' },
    { key: 'TempMax', label: 'Maximum Temperature' },
    { key: 'RainMm', label: 'Rainfall (mm)' },
    { key: 'SnowCm', label: 'Snowfall (cm)' },
    { key: 'WindMax', label: 'Maximum Wind Speed' },
    { key: 'SunHours', label: 'Sunshine Hours' },
  ];

  forecastImpact: ForecastImpactResponse | null = null;
  forecastImpactLoading = false;
  forecastImpactError = '';

  factorContribution: FactorContributionResponse | null = null;
  factorContributionLoading = false;
  factorContributionError = '';

  featureImportance: FeatureImportanceResponse | null = null;
  featureImportanceLoading = false;
  featureImportanceError = '';

  selectedMetric = 'TempAvg';

  activeTab: AnalysisTab = 'correlation';

  setActiveTab(tab: AnalysisTab): void {
    this.activeTab = tab;
  }


  loadForecastImpact(): void {
    if (!this.canLoad()) {
      return;
    }
  
    this.forecastImpactLoading = true;
    this.forecastImpactError = '';
  
    const params = new HttpParams()
      .set('productid', this.selectedProductId)
      .set('channelid', this.selectedChannelId)
      .set('locationid', this.selectedLocationId)
      .set(
        'period',
        this.selectedPeriod === 'daily'
          ? 'Daily'
          : this.selectedPeriod === 'weekly'
          ? 'Weekly'
          : 'Monthly'
      )
      .set('level', '111')
      .set('scenario_id', '1');
  
    this.http
      .get<ForecastImpactResponse>(
        `${this.apiBase}/api/forecast/backtest-compare-selected`,
        { params }
      )
      .subscribe({
        next: (response) => {
          this.forecastImpact = response;
          this.forecastImpactLoading = false;
        },
  
        error: (error) => {
          console.error('Forecast impact error:', error);
  
          this.forecastImpactError =
            error?.error?.detail ||
            'Unable to calculate forecast impact.';
  
          this.forecastImpactLoading = false;
        },
      });
  }

  loadFactorContribution(): void {
    if (!this.canLoad()) {
      return;
    }
  
    this.factorContributionLoading = true;
    this.factorContributionError = '';
    this.factorContribution = null;
  
    const params = new HttpParams()
      .set('productid', this.selectedProductId)
      .set('channelid', this.selectedChannelId)
      .set('locationid', this.selectedLocationId)
      .set(
        'period',
        this.selectedPeriod === 'daily'
          ? 'Daily'
          : this.selectedPeriod === 'weekly'
          ? 'Weekly'
          : 'Monthly'
      )
      .set('level', '111')
      .set('scenario_id', '1');
  
    this.http
      .get<FactorContributionResponse>(
        `${this.apiBase}/api/forecast/backtest-contribution-selected`,
        { params }
      )
      .subscribe({
        next: (response) => {
          this.factorContribution = response;
          this.factorContributionLoading = false;
        },
  
        error: (error) => {
          console.error('Factor contribution error:', error);
  
          this.factorContributionError =
            error?.error?.detail ||
            'Unable to calculate factor contribution.';
  
          this.factorContributionLoading = false;
        },
      });
  }

  loadFeatureImportance(): void {
    if (!this.canLoad()) {
      return;
    }
  
    this.featureImportanceLoading = true;
    this.featureImportanceError = '';
    this.featureImportance = null;
  
    const params = new HttpParams()
      .set('productid', this.selectedProductId)
      .set('channelid', this.selectedChannelId)
      .set('locationid', this.selectedLocationId)
      .set(
        'period',
        this.selectedPeriod === 'daily'
          ? 'Daily'
          : this.selectedPeriod === 'weekly'
          ? 'Weekly'
          : 'Monthly'
      )
      .set('level', '111')
      .set('scenario_id', '1');
  
    this.http
      .get<FeatureImportanceResponse>(
        `${this.apiBase}/api/forecast/feature-importance-selected`,
        { params }
      )
      .subscribe({
        next: (response) => {
          this.featureImportance = response;
          this.featureImportanceLoading = false;
        },
  
        error: (error) => {
          console.error('Feature importance error:', error);
  
          this.featureImportanceError =
            error?.error?.detail ||
            'Unable to calculate feature importance.';
  
          this.featureImportanceLoading = false;
        },
      });
  }


  // Data from backend
  weatherSales: WeatherSalesPoint[] = [];
  correlations: CorrelationResponse = {
    TempAvg: null,
    TempMin: null,
    TempMax: null,
    RainMm: null,
    SnowCm: null,
    WindMax: null,
    SunHours: null,
  };

  // Derived for UI
  scatterPoints: ScatterPoint[] = [];
  xMin = 0;
  xMax = 0;
  yMin = 0;
  yMax = 0;
  xMid = 0;
  yMid = 0;
  xLabels: string[] = [];
  yLabels: string[] = [];

  // SVG layout
  svgWidth = 420;
  svgHeight = 260;
  plotPaddingLeft = 50;
  plotPaddingRight = 10;
  plotPaddingTop = 10;
  plotPaddingBottom = 40;

  // Correlation for selected metric
  corrSelected: number | null = null;
  corrPercent = 50; // 0–100 position on -1..1 scale
  corrDescription = '';

  // UI state
  loading = false;
  errorMessage = '';

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadProducts();
    this.loadChannels();
    this.loadLocations();
  }

  // ----------------------------
  // Load dropdowns
  // ----------------------------
  private loadProducts(): void {
    this.http.get<Product[]>(`${this.apiBase}/api/products`).subscribe({
      next: (rows) => {
        this.products = rows || [];
      },
      error: (err) => {
        console.error('Error loading products', err);
      },
    });
  }

  private loadChannels(): void {
    this.http.get<Channel[]>(`${this.apiBase}/api/channels`).subscribe({
      next: (rows) => {
        this.channels = rows || [];
      },
      error: (err) => {
        console.error('Error loading channels', err);
      },
    });
  }

  private loadLocations(): void {
    this.http.get<Location[]>(`${this.apiBase}/api/locations`).subscribe({
      next: (rows) => {
        this.locations = rows || [];
      },
      error: (err) => {
        console.error('Error loading locations', err);
      },
    });
  }

  // ----------------------------
  // Event handlers
  // ----------------------------


  onSelectionChange(): void {
    this.forecastImpact = null;
    this.forecastImpactError = '';
  
    this.factorContribution = null;
    this.factorContributionError = '';
  
    this.featureImportance = null;
    this.featureImportanceError = '';
  
    if (this.canLoad()) {
      this.loadAll();
    }
  }

  onMetricChange(): void {
    // Just update selected metric correlation + scatter labels
    this.updateCorrelationUI();
  }

  onReloadClick(): void {
    if (this.canLoad()) {
      this.loadAll();
    }
  }

  // made public so template can call it
  canLoad(): boolean {
    return (
      !!this.selectedProductId &&
      !!this.selectedChannelId &&
      !!this.selectedLocationId &&
      !!this.selectedPeriod &&
      !!this.selectedMetric
    );
  }

  // ----------------------------
  // Load weather + sales + correlations
  // ----------------------------
  private loadAll(): void {
    this.loading = true;
    this.errorMessage = '';
    this.weatherSales = [];
    this.scatterPoints = [];

    const paramsBase = new HttpParams()
      .set('productId', this.selectedProductId)
      .set('channelId', this.selectedChannelId)
      .set('locationId', this.selectedLocationId)
      .set('period', this.selectedPeriod);

    // 1) Load weather-sales series
    const wsParams = paramsBase.set('metric', this.selectedMetric);

    this.http
      .get<WeatherSalesPoint[]>(`${this.apiBase}/api/weather-sales`, { params: wsParams })
      .subscribe({
        next: (rows) => {
          this.weatherSales = rows || [];
          this.buildScatter();

          // 2) Load correlations after we have scatter
          this.http
            .get<CorrelationResponse>(`${this.apiBase}/api/weather-correlations`, {
              params: paramsBase,
            })
            .subscribe({
              next: (corr) => {
                this.correlations = corr || this.correlations;
                this.updateCorrelationUI();
                this.loading = false;
              },
              error: (err) => {
                console.error('Error loading correlations', err);
                this.loading = false;
                this.updateCorrelationUI(); // at least update with whatever we have
              },
            });
        },
        error: (err) => {
          console.error('Error loading weather-sales data', err);
          this.loading = false;
          this.errorMessage = 'Failed to load weather and sales data.';
        },
      });
  }

  // ----------------------------
  // Build scatter plot
  // ----------------------------
  private buildScatter(): void {
    this.scatterPoints = [];
    this.xLabels = [];
    this.yLabels = [];

    if (!this.weatherSales.length) {
      this.xMin = this.xMax = this.yMin = this.yMax = 0;
      return;
    }

    const xs = this.weatherSales.map((p) => p.weatherMetric);
    const ys = this.weatherSales.map((p) => p.quantity);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const xRange = maxX - minX || 1; // avoid 0
    const yRange = maxY - minY || 1;

    this.xMin = minX;
    this.xMax = maxX;
    this.yMin = minY;
    this.yMax = maxY;

    this.xMid = minX + xRange / 2;
    this.yMid = minY + yRange / 2;

    const innerWidth = this.svgWidth - this.plotPaddingLeft - this.plotPaddingRight;
    const innerHeight = this.svgHeight - this.plotPaddingTop - this.plotPaddingBottom;

    const xLabelMin = this.roundNice(minX);
    const xLabelMid = this.roundNice(this.xMid);
    const xLabelMax = this.roundNice(maxX);
    const yLabelMin = this.roundNice(minY);
    const yLabelMid = this.roundNice(this.yMid);
    const yLabelMax = this.roundNice(maxY);

    this.xLabels = [String(xLabelMin), String(xLabelMid), String(xLabelMax)];
    this.yLabels = [String(yLabelMin), String(yLabelMid), String(yLabelMax)];

    this.scatterPoints = this.weatherSales.map((p) => {
      const xNorm = (p.weatherMetric - minX) / xRange;
      const yNorm = (p.quantity - minY) / yRange;

      const xPx = this.plotPaddingLeft + xNorm * innerWidth;
      const yPx =
        this.plotPaddingTop + (1 - yNorm) * innerHeight; // invert Y for SVG

      return {
        x: xPx,
        y: yPx,
        date: p.date,
        weatherMetric: p.weatherMetric,
        quantity: p.quantity,
      };
    });
  }

  // ----------------------------
  // Correlation UI
  // ----------------------------
  private updateCorrelationUI(): void {
    const r = this.correlations?.[this.selectedMetric] ?? null;
    this.corrSelected = Number.isFinite(r as number) ? (r as number) : null;

    if (this.corrSelected == null) {
      this.corrPercent = 50;
      this.corrDescription = 'No correlation data';
      return;
    }

    const rClamped = Math.max(-1, Math.min(1, this.corrSelected));
    this.corrPercent = ((rClamped + 1) / 2) * 100;

    if (rClamped > 0.5) {
      this.corrDescription = 'Strong positive correlation';
    } else if (rClamped > 0.2) {
      this.corrDescription = 'Moderate positive correlation';
    } else if (rClamped > -0.2) {
      this.corrDescription = 'Weak or no correlation';
    } else if (rClamped > -0.5) {
      this.corrDescription = 'Moderate negative correlation';
    } else {
      this.corrDescription = 'Strong negative correlation';
    }
  }

  // ----------------------------
  // Small helpers
  // ----------------------------
  getMetricLabel(key: string): string {
    const opt = this.metricOptions.find((m) => m.key === key);
    return opt ? opt.label : key;
  }

  getFeatureLabel(feature: string): string {
    const labels: Record<string, string> = {
      lag_1: 'Previous Period Demand',
      lag_4: 'Demand 4 Periods Ago',
      lag_13: 'Demand 13 Periods Ago',
  
      roll_mean_4: '4-Period Average Demand',
      roll_mean_13: '13-Period Average Demand',
  
      roll_min_4: '4-Period Minimum Demand',
      roll_min_13: '13-Period Minimum Demand',
  
      roll_max_4: '4-Period Maximum Demand',
      roll_max_13: '13-Period Maximum Demand',
  
      roll_std_4: '4-Period Demand Variation',
      roll_std_13: '13-Period Demand Variation',
  
      NetPrice: 'Net Price',
      ListPrice: 'List Price',
      DiscountRate: 'Discount Rate',
  
      TavgC: 'Average Temperature',
      TminC: 'Minimum Temperature',
      TmaxC: 'Maximum Temperature',
      PrecipMM: 'Precipitation',
      WindMaxMS: 'Maximum Wind Speed',
      SunHours: 'Sunshine Hours',
  
      PromoFlag: 'Promotion Active',
      PromoDepth: 'Promotion Depth',
      PromoUplift: 'Promotion Uplift',
  
      year: 'Year',
      month: 'Month',
      weekofyear: 'Week of Year',
      dow: 'Day of Week',
      doy: 'Day of Year',
    };
  
    return labels[feature] || feature;
  }

  private roundNice(v: number): number {
    if (!Number.isFinite(v)) return 0;
    // Simple rounding to 1 decimal if needed
    const abs = Math.abs(v);
    if (abs < 10) {
      return Math.round(v * 10) / 10;
    }
    return Math.round(v);
  }

  // For template: axis positions
  get centerX(): number {
    if (!this.weatherSales.length) {
      return this.plotPaddingLeft + (this.svgWidth - this.plotPaddingLeft - this.plotPaddingRight) / 2;
    }
    const innerWidth = this.svgWidth - this.plotPaddingLeft - this.plotPaddingRight;
    const xRange = this.xMax - this.xMin || 1;
    const xNorm = (this.xMid - this.xMin) / xRange;
    return this.plotPaddingLeft + xNorm * innerWidth;
  }

  get centerY(): number {
    if (!this.weatherSales.length) {
      return this.plotPaddingTop + (this.svgHeight - this.plotPaddingTop - this.plotPaddingBottom) / 2;
    }
    const innerHeight = this.svgHeight - this.plotPaddingTop - this.plotPaddingBottom;
    const yRange = this.yMax - this.yMin || 1;
    const yNorm = (this.yMid - this.yMin) / yRange;
    return this.plotPaddingTop + (1 - yNorm) * innerHeight;
  }
}
