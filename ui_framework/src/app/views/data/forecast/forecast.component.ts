import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  CardFooterComponent,
  CardHeaderComponent,
  ColComponent,
  RowComponent,
  TableDirective,
  TextColorDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';
import { firstValueFrom } from 'rxjs';
import { ScenarioService } from '../../../services/scenario.service';



interface SavedSearch {
  id?: number;
  name: string;
  query: string;
  created_at?: string;
}

interface IForecastRow {
  Level: string;
  Model: string;
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  EndDate: string;
  Period: string;
  ForecastQty: string | number;
  UOM: string;
  NetPrice: string | number;
  ListPrice: string | number;
  Method: string;
}

@Component({
  standalone: true,
  selector: 'app-forecast',
  templateUrl: 'forecast.component.html',
  styleUrls: ['forecast.component.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,
    TextColorDirective,
    CardComponent,
    CardBodyComponent,
    CardHeaderComponent,
    CardFooterComponent,
    RowComponent,
    ColComponent,
    ButtonDirective,
    IconDirective,
    TableDirective
  ]
})
export class ForecastComponent implements OnInit {
  private http: HttpClient = inject(HttpClient);

  // Public so HTML can use it
  readonly scenarioService = inject(ScenarioService);

  private readonly API = 'http://127.0.0.1:8000/api';
  private readonly DB_SCHEMA = 'planwise_fresh_produce';

  /** Baseline vs Feat */
  public variant = new FormControl<'baseline' | 'feat'>('baseline');

  /** Period */
  public period = new FormControl<'Daily' | 'Weekly' | 'Monthly'>('Weekly');

  /** Typed search */
  public searchField = new FormControl<'ProductID' | 'ChannelID' | 'LocationID'>('ProductID');
  public searchTerm = new FormControl('');

  /** Optional typed filters */
  public modelTerm = new FormControl('');
  public methodTerm = new FormControl('');

  /** Saved searches */
  public savedSearches: SavedSearch[] = [];
  public selectedSavedIndex = new FormControl<number>(-1);

  /** Result rows (server-paged) */
  public rows: IForecastRow[] = [];
  public totalRows = 0;

  /** UI state */
  public loading = false;
  public errorMessage: string | null = null;

  /** Pagination (server-side) */
  public currentPage = 1;
  public pageSize = 200;

  ngOnInit(): void {
    this.refreshSavedSearches();
    this.clearRows();

    // auto-search while typing
    this.searchTerm.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });

    this.searchField.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });

    this.variant.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });

    this.period.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });

    this.modelTerm.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });

    this.methodTerm.valueChanges.subscribe(() => {
      this.currentPage = 1;
      this.runSearch().catch(() => {});
    });
  }

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.totalRows / this.pageSize));
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  private clearRows(): void {
    this.rows = [];
    this.totalRows = 0;
    this.currentPage = 1;
  }

  private baseParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', this.currentScenarioId);
  }

  refreshSavedSearches(): void {
    const params = new HttpParams().set('db_schema', this.DB_SCHEMA);

    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`, { params }).subscribe({
      next: rows => {
        this.savedSearches = rows || [];
      },
      error: e => {
        console.error('saved-searches failed', e);
      }
    });
  }

  /** Optional: load saved query text into the input */
  useSavedQuery(): void {
    this.errorMessage = null;

    const idx = this.selectedSavedIndex.value ?? -1;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMessage = 'Please choose a saved search.';
      return;
    }

    this.searchTerm.setValue(this.savedSearches[idx].query || '');
  }

  /** Main typed search */
  async runSearch(): Promise<void> {
    this.errorMessage = null;

    const term = (this.searchTerm.value || '').trim();
    if (!term) {
      this.clearRows();
      return;
    }

    const variant = this.variant.value || 'baseline';
    const field = this.searchField.value || 'ProductID';
    const period = this.period.value || 'Weekly';
    const offset = (this.currentPage - 1) * this.pageSize;

    this.loading = true;

    try {
      let params = this.baseParams()
        .set('variant', variant)
        .set('field', field)
        .set('term', term)
        .set('period', period)
        .set('limit', this.pageSize)
        .set('offset', offset);

      const model = (this.modelTerm.value || '').trim();
      const method = (this.methodTerm.value || '').trim();

      if (model) {
        params = params.set('model', model);
      }

      if (method) {
        params = params.set('method', method);
      }

      const res = await firstValueFrom(
        this.http.get<any>(`${this.API}/forecast/search`, { params })
      );

      this.totalRows = Number(res?.count ?? 0);

      this.rows = (res?.rows ?? []).map((r: any) => ({
        Level: String(r.Level ?? ''),
        Model: String(r.Model ?? ''),
        ProductID: String(r.ProductID ?? ''),
        ChannelID: String(r.ChannelID ?? ''),
        LocationID: String(r.LocationID ?? ''),
        StartDate: String(r.StartDate ?? ''),
        EndDate: String(r.EndDate ?? ''),
        Period: String(r.Period ?? ''),
        ForecastQty: r.ForecastQty ?? '',
        UOM: String(r.UOM ?? ''),
        NetPrice: r.NetPrice ?? '',
        ListPrice: r.ListPrice ?? '',
        Method: String(r.Method ?? '')
      }));

      const tp = this.totalPages;
      if (this.currentPage > tp) {
        this.currentPage = tp;
      }
    } catch (e: any) {
      console.error('Forecast search failed', e);
      this.errorMessage = e?.error?.detail || e?.message || 'Failed to load forecast.';
      this.clearRows();
    } finally {
      this.loading = false;
    }
  }

  setPage(p: number): void {
    const total = this.totalPages;
    this.currentPage = Math.min(Math.max(1, p), total);
    this.runSearch().catch(() => {});
  }

  visiblePages(): number[] {
    const total = this.totalPages || 1;
    const current = Math.min(Math.max(this.currentPage, 1), total);
    const windowSize = 5;

    if (total <= windowSize) {
      return Array.from({ length: total }, (_, i) => i + 1);
    }

    let start = current - Math.floor(windowSize / 2);
    let end = current + Math.floor(windowSize / 2);

    if (start < 1) {
      start = 1;
      end = windowSize;
    }

    if (end > total) {
      end = total;
      start = total - windowSize + 1;
    }

    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }

  clearSearch(): void {
    this.searchTerm.setValue('');
    this.modelTerm.setValue('');
    this.methodTerm.setValue('');
    this.clearRows();
  }

  exportToCSV(): void {
    if (!this.rows.length) return;

    const header = [
      'Level',
      'Model',
      'ProductID',
      'ChannelID',
      'LocationID',
      'StartDate',
      'EndDate',
      'Period',
      'ForecastQty',
      'UOM',
      'NetPrice',
      'ListPrice',
      'Method'
    ];

    const body = this.rows.map(r =>
      header.map(h => `"${String((r as any)[h] ?? '').replace(/"/g, '""')}"`).join(',')
    );

    const csv = [header.join(','), ...body].join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `Forecast_s${this.currentScenarioId}_${this.variant.value}_${this.period.value}_page${this.currentPage}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}