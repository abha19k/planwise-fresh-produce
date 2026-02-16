// src/app/views/data/supply-data/supply-data.component.ts

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';

import {
  CardComponent,
  CardHeaderComponent,
  CardBodyComponent,
  RowComponent,
  ColComponent,
  TableDirective,
} from '@coreui/angular';

interface BomItem {
  ProductID: string;
  ProductName: string;
  ItemID: string;
  ItemName: string;
  ItemQty: number;
  UnitofMeasurement: string;
}

interface LotSizeItem {
  ItemID: string;
  ItemName?: string | null;
  LotSize: number;
  UnitofMeasurement?: string | null;
}

interface SafetyStockItem {
  ItemID: string;
  ItemName?: string | null;
  SafetyStockRule?: string | null;
  DaysOfSafetyStock: number;
}

interface InventoryItem {
  ItemID: string;
  LocationID: string;
  Qty: number;
  UnitofMeasurement?: string | null;
}

interface SourcingOrderItem {
  ItemID: string;
  LocationID: string;
  ArrivalDate: string;
  Qty: number;
  UnitofMeasurement?: string | null;
}

interface ForecastRow {
  ProductID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Qty: number;
}

interface ItemForecastRow {
  ItemID: string;
  ItemName: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Qty: number;
  UnitofMeasurement: string;
}

interface RecommendedSourcingRow {
  ItemID: string;
  ItemName: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Qty: number;
  UnitofMeasurement: string;
}

interface InventoryProfilePoint {
  ItemID: string;
  LocationID: string;
  Date: string;
  DemandQty: number;
  SafetyStockQty: number;
  EndInventoryQty: number;
  InboundConfirmedQty: number;
  RecommendedOrderQty: number;
}

interface ItemOption {
  ItemID: string;
  ItemName?: string | null;
}

interface LocationOption {
  LocationID: string;
}

type PeriodType = 'daily' | 'weekly' | 'monthly';

type SupplyTab =
  | 'bom'
  | 'lotsize'
  | 'safetystock'
  | 'inventory'
  | 'sourcingorder'
  | 'forecast'
  | 'itemforecast'
  | 'recommended'
  | 'invProfile';

@Component({
  selector: 'app-supply-data',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    RowComponent,
    ColComponent,
    TableDirective,
  ],
  templateUrl: './supply-data.component.html',
})
export class SupplyDataComponent implements OnInit {
  private readonly baseUrl = 'http://localhost:8000';

  activeTab: SupplyTab = 'bom';

  // raw data
  bomItems: BomItem[] = [];
  lotSizeItems: LotSizeItem[] = [];
  safetyStockItems: SafetyStockItem[] = [];
  inventoryItems: InventoryItem[] = [];
  sourcingOrderItems: SourcingOrderItem[] = [];

  // forecast data (product/location)
  forecastPeriod: PeriodType = 'daily';
  forecastRows: ForecastRow[] = [];

  // item-level forecast data
  itemForecastPeriod: PeriodType = 'daily';
  itemForecastRows: ItemForecastRow[] =[];

  // recommended sourcing orders
  recommendedPeriod: PeriodType = 'daily';
  recommendedRows: RecommendedSourcingRow[] = [];

  // 🔹 Inventory profile (item × location)
  inventoryProfile: InventoryProfilePoint[] = [];
  selectedItemId = '';
  selectedLocationId = '';

  // options for dropdowns (used in HTML)
  itemsForSelect: ItemOption[] = [];
  locationsForSelect: LocationOption[] = [];

  // simple SVG chart geometry
  safetyPoints = '';
  inventoryPoints = '';
  chartWidth = 560;
  chartHeight = 220;
  chartLeft = 40;
  chartTop = 20;

  // loading & error
  loading = false;
  errorMessage = '';

  tabs: { id: SupplyTab; label: string }[] = [
    { id: 'bom',           label: 'Bill of Materials (BOM)' },
    { id: 'lotsize',       label: 'Lot Size' },
    { id: 'safetystock',   label: 'Safety Stock' },
    { id: 'inventory',     label: 'Inventory' },
    { id: 'sourcingorder', label: 'Confirmed Sourcing Orders' },
    { id: 'forecast',      label: 'Product-Level Forecast' },
    { id: 'itemforecast',  label: 'Item Level Forecast' },
    { id: 'recommended',   label: 'Recommended Sourcing Order' },
    { id: 'invProfile',    label: 'Inventory Profile' },
  ];

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadAll();
  }

  setTab(tab: SupplyTab): void {
    this.activeTab = tab;

    if (tab === 'forecast' && this.forecastRows.length === 0) {
      this.loadForecast();
    }

    if (tab === 'itemforecast' && this.itemForecastRows.length === 0) {
      this.loadItemForecast();
    }

    if (tab === 'recommended' && this.recommendedRows.length === 0) {
      this.loadRecommended();
    }

    if (tab === 'invProfile') {
      this.loadInventoryProfile();
    }
  }

  onForecastPeriodChange(period: PeriodType): void {
    this.forecastPeriod = period;
    if (this.activeTab === 'forecast') {
      this.loadForecast();
    }
  }

  onItemForecastPeriodChange(period: PeriodType): void {
    this.itemForecastPeriod = period;
    if (this.activeTab === 'itemforecast') {
      this.loadItemForecast();
    }
  }

  onRecommendedPeriodChange(period: PeriodType): void {
    this.recommendedPeriod = period;
    if (this.activeTab === 'recommended') {
      this.loadRecommended();
    }
  }

  // ================================================================
  // Initial load (BOM, lot size, safety stock, inventory, sourcing)
  // ================================================================
  public readonly Math = Math;
  private loadAll(): void {
    this.loading = true;
    this.errorMessage = '';

    this.http.get<BomItem[]>(`${this.baseUrl}/api/bom`).subscribe({
      next: (data) => {
        this.bomItems = data;
      },
      error: (err) => (this.errorMessage ||= `Failed to load BOM: ${err.message ?? err}`),
    });

    this.http.get<LotSizeItem[]>(`${this.baseUrl}/api/lotsize`).subscribe({
      next: (data) => {
        this.lotSizeItems = data;
        this.updateItemOptions();
      },
      error: (err) => (this.errorMessage ||= `Failed to load LotSize: ${err.message ?? err}`),
    });

    this.http.get<SafetyStockItem[]>(`${this.baseUrl}/api/safetystock`).subscribe({
      next: (data) => (this.safetyStockItems = data),
      error: (err) => (this.errorMessage ||= `Failed to load SafetyStock: ${err.message ?? err}`),
    });

    this.http.get<InventoryItem[]>(`${this.baseUrl}/api/inventory`).subscribe({
      next: (data) => {
        this.inventoryItems = data;
        this.updateLocationOptions();
      },
      error: (err) => (this.errorMessage ||= `Failed to load Inventory: ${err.message ?? err}`),
    });

    this.http.get<SourcingOrderItem[]>(`${this.baseUrl}/api/sourcingorder`).subscribe({
      next: (data) => {
        this.sourcingOrderItems = data;
        this.loading = false;
      },
      error: (err) => {
        this.errorMessage ||= `Failed to load SourcingOrder: ${err.message ?? err}`;
        this.loading = false;
      },
    });
  }

  private updateItemOptions(): void {
    const seen = new Set<string>();
    const opts: ItemOption[] = [];

    for (const row of this.lotSizeItems) {
      if (!seen.has(row.ItemID)) {
        seen.add(row.ItemID);
        opts.push({
          ItemID: row.ItemID,
          ItemName: row.ItemName ?? undefined,
        });
      }
    }

    this.itemsForSelect = opts.sort((a, b) => a.ItemID.localeCompare(b.ItemID));

    if (!this.selectedItemId && this.itemsForSelect.length > 0) {
      this.selectedItemId = this.itemsForSelect[0].ItemID;
    }
  }

  private updateLocationOptions(): void {
    const seen = new Set<string>();
    const opts: LocationOption[] = [];

    for (const inv of this.inventoryItems) {
      if (!seen.has(inv.LocationID)) {
        seen.add(inv.LocationID);
        opts.push({ LocationID: inv.LocationID });
      }
    }

    this.locationsForSelect = opts.sort((a, b) => a.LocationID.localeCompare(b.LocationID));

    if (!this.selectedLocationId && this.locationsForSelect.length > 0) {
      this.selectedLocationId = this.locationsForSelect[0].LocationID;
    }
  }

  // =========================
  // Forecast data loaders
  // =========================
  private loadForecast(): void {
    this.loading = true;
    this.errorMessage = '';

    this.http
      .get<ForecastRow[]>(`${this.baseUrl}/api/supply-forecast`, {
        params: { period: this.forecastPeriod },
      })
      .subscribe({
        next: (rows) => {
          this.forecastRows = rows;
          this.loading = false;
        },
        error: (err) => {
          this.errorMessage ||= `Failed to load Forecast: ${err.message ?? err}`;
          this.loading = false;
        },
      });
  }

  private loadItemForecast(): void {
    this.loading = true;
    this.errorMessage = '';

    this.http
      .get<ItemForecastRow[]>(`${this.baseUrl}/api/item-forecast`, {
        params: { period: this.itemForecastPeriod },
      })
      .subscribe({
        next: (rows) => {
          this.itemForecastRows = rows;
          this.loading = false;
        },
        error: (err) => {
          this.errorMessage ||= `Failed to load Item Level Forecast: ${err.message ?? err}`;
          this.loading = false;
        },
      });
  }

  private loadRecommended(): void {
    this.loading = true;
    this.errorMessage = '';

    this.http
      .get<RecommendedSourcingRow[]>(`${this.baseUrl}/api/recommended-sourcing`, {
        params: { period: this.recommendedPeriod },
      })
      .subscribe({
        next: (rows) => {
          this.recommendedRows = rows;
          this.loading = false;
        },
        error: (err) => {
          this.errorMessage ||= `Failed to load Recommended Sourcing Orders: ${err.message ?? err}`;
          this.loading = false;
        },
      });
  }

  // =========================
  // Inventory Profile (tab)
  // =========================

  // =========================
// Inventory Profile (tab)
// =========================
yTicks: number[] = [];   // Y-axis tick values

loadInventoryProfile(): void {
  if (this.activeTab !== 'invProfile') return;

  if (!this.selectedItemId || !this.selectedLocationId) {
    this.inventoryProfile = [];
    this.safetyPoints = '';
    this.inventoryPoints = '';
    this.yTicks = [];
    return;
  }

  this.loading = true;
  this.errorMessage = '';

  const params = new HttpParams()
    .set('itemId', this.selectedItemId)
    .set('locationId', this.selectedLocationId)
    .set('period', 'daily')
    .set('method', 'XGBoost');

  this.http
    .get<InventoryProfilePoint[]>(`${this.baseUrl}/api/inventory-profile`, { params })
    .subscribe({
      next: (rows) => {
        this.inventoryProfile = rows || [];
        this.updateInventoryChartPoints();
        this.loading = false;
      },
      error: (err) => {
        this.errorMessage ||= `Failed to load Inventory Profile: ${err.message ?? err}`;
        this.inventoryProfile = [];
        this.safetyPoints = '';
        this.inventoryPoints = '';
        this.yTicks = [];
        this.loading = false;
      },
    });
}

// =========================
// Build SVG points + ticks
// =========================
private updateInventoryChartPoints(): void {
  const pts = this.inventoryProfile;
  if (!pts || !pts.length) {
    this.safetyPoints = '';
    this.inventoryPoints = '';
    this.yTicks = [];
    return;
  }

  // Determine max numeric value
  const maxVal = Math.max(
    ...pts.map(p => Math.max(p.SafetyStockQty, p.EndInventoryQty)),
    1
  );

  // Create 5 ticks → e.g., 0%, 25%, 50%, 75%, 100%
  this.yTicks = [];
  for (let i = 0; i <= 4; i++) {
    this.yTicks.push(Math.round((maxVal / 4) * i));
  }

  const n = pts.length;
  const xStep = this.chartWidth / Math.max(n - 1, 1);

  const makePoints = (getVal: (p: InventoryProfilePoint) => number) =>
    pts
      .map((p, idx) => {
        const x = this.chartLeft + idx * xStep;

        const val = getVal(p);
        const y =
          this.chartTop +
          (1 - val / maxVal) * this.chartHeight; // top=high qty, bottom=low qty

        return `${x},${y}`;
      })
      .join(' ');

  this.safetyPoints = makePoints(p => p.SafetyStockQty);
  this.inventoryPoints = makePoints(p => p.EndInventoryQty);
}

// =========================
// Helper functions for HTML
// =========================
getYForValue(value: number): number {
  if (!this.yTicks.length) return this.chartTop + this.chartHeight;

  const maxVal = this.yTicks[this.yTicks.length - 1] || 1;
  return this.chartTop + (1 - value / maxVal) * this.chartHeight;
}

getXForIndex(index: number, total: number): number {
  if (total <= 1) return this.chartLeft;
  const xStep = this.chartWidth / (total - 1);
  return this.chartLeft + index * xStep;
}
}
