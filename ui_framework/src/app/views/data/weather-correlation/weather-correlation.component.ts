// src/app/views/data/weather-correlation/weather-correlation.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';

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
  [key: string]: number | null;
}

@Component({
  selector: 'app-weather-correlation',
  standalone: true,
  templateUrl: './weather-correlation.component.html',
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
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
  selectedPeriod: PeriodType = 'daily';

  metricOptions = [
    { key: 'TempAvg', label: 'Average Temperature' },
    { key: 'TempMin', label: 'Minimum Temperature' },
    { key: 'TempMax', label: 'Maximum Temperature' },
    { key: 'RainMm', label: 'Rainfall (mm)' },
    { key: 'SnowCm', label: 'Snowfall (cm)' },
  ];
  selectedMetric = 'TempAvg';

  // Data from backend
  weatherSales: WeatherSalesPoint[] = [];
  correlations: CorrelationResponse = {
    TempAvg: null,
    TempMin: null,
    TempMax: null,
    RainMm: null,
    SnowCm: null,
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
