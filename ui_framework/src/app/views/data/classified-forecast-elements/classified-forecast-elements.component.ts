import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
import { Router } from '@angular/router';
import {
  CardComponent,
  CardHeaderComponent,
  CardBodyComponent,
  CardFooterComponent,
  RowComponent,
  ColComponent,
  ButtonDirective,
  TextColorDirective,
  TableDirective
} from '@coreui/angular';
import { IconDirective } from '@coreui/icons-angular';

import { ScenarioService } from '../../../services/scenario.service';

type PeriodView = 'Daily' | 'Weekly' | 'Monthly';
type PeriodSlug = 'daily' | 'weekly' | 'monthly';

interface RowVM {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  Period: PeriodView;
  adi: number | null;
  cv2: number | null;
  category: string;
  algorithm: string;
  isActive?: boolean;
  created_at?: string;
  scenario_id?: number;
}

@Component({
  standalone: true,
  selector: 'app-classified-forecast-elements',
  templateUrl: './classified-forecast-elements.component.html',
  styleUrls: ['./classified-forecast-elements.component.scss'],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,
    TextColorDirective,
    TableDirective,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    CardFooterComponent,
    RowComponent,
    ColComponent,
    ButtonDirective,
    IconDirective
  ]
})
export class ClassifiedForecastElementsComponent implements OnInit {
  private http = inject(HttpClient);
  private router = inject(Router);

  readonly scenarioService = inject(ScenarioService);

  readonly API = 'http://127.0.0.1:8000/api';
  readonly DB_SCHEMA = 'planwise_fresh_produce';

  period = new FormControl<PeriodView>('Daily', { nonNullable: true });
  includeInactive = new FormControl<boolean>(false, { nonNullable: true });

  rows: RowVM[] = [];
  loading = false;
  errorMsg: string | null = null;

  sortColumn: keyof RowVM | '' = '';
  sortAsc = true;

  ngOnInit(): void {
    this.loadRows();

    this.period.valueChanges.subscribe(() => this.loadRows());
    this.includeInactive.valueChanges.subscribe(() => this.loadRows());
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  private slug(): PeriodSlug {
    const p = this.period.value;
    return p === 'Weekly' ? 'weekly' : p === 'Monthly' ? 'monthly' : 'daily';
  }

  private baseParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', this.currentScenarioId);
  }

  private toVM(r: any): RowVM {
    const slug = this.slug();
    const periodView: PeriodView =
      (r.Period as PeriodView) ??
      (slug === 'weekly' ? 'Weekly' : slug === 'monthly' ? 'Monthly' : 'Daily');

    return {
      ProductID: String(r.ProductID ?? ''),
      ChannelID: String(r.ChannelID ?? ''),
      LocationID: String(r.LocationID ?? ''),
      Period: periodView,
      adi: r.ADI ?? r.adi ?? null,
      cv2: r.CV2 ?? r.cv2 ?? null,
      category: String(r.Category ?? r.category ?? ''),
      algorithm: String(r.Algorithm ?? r.algorithm ?? ''),
      isActive: r.IsActive ?? r.is_active ?? undefined,
      created_at:
        r.UpdatedAt ??
        r.CreatedAt ??
        r.created_at ??
        r.updated_at ??
        undefined,
      scenario_id: r.scenario_id ?? undefined
    };
  }

  loadRows(): void {
    this.loading = true;
    this.errorMsg = null;
    this.rows = [];

    const params = this.baseParams()
      .set('period', this.slug())
      .set('include_inactive', String(this.includeInactive.value))
      .set('limit', '20000')
      .set('offset', '0');

    this.http.get<any[]>(`${this.API}/classify/saved`, { params }).subscribe({
      next: res => {
        this.rows = (res || []).map(x => this.toVM(x));

        if (this.rows.some(r => !!r.created_at)) {
          this.sortColumn = 'created_at';
          this.sortAsc = false;
          this.sortRows('created_at', false);
        } else {
          this.sortColumn = 'ProductID';
          this.sortAsc = true;
          this.sortRows('ProductID', true);
        }
      },
      error: e => {
        this.errorMsg = e?.error?.detail || e?.message || 'Failed to load saved classified results.';
      },
      complete: () => {
        this.loading = false;
      }
    });
  }

  refresh(): void {
    this.loadRows();
  }

  sortBy(col: keyof RowVM): void {
    if (this.sortColumn === col) {
      this.sortAsc = !this.sortAsc;
    } else {
      this.sortColumn = col;
      this.sortAsc = true;
    }

    this.sortRows(col, this.sortAsc);
  }

  private sortRows(col: keyof RowVM, asc: boolean): void {
    const dir = asc ? 1 : -1;

    this.rows.sort((a, b) => {
      const av = a[col];
      const bv = b[col];

      if (col === 'adi' || col === 'cv2') {
        const na = av == null ? Number.POSITIVE_INFINITY : Number(av);
        const nb = bv == null ? Number.POSITIVE_INFINITY : Number(bv);
        return (na - nb) * dir;
      }

      if (col === 'created_at') {
        const da = av ? new Date(String(av)).getTime() : 0;
        const db = bv ? new Date(String(bv)).getTime() : 0;
        return (da - db) * dir;
      }

      return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
    });
  }

  exportCSV(): void {
    if (!this.rows.length) return;

    const header = [
      'ScenarioID',
      'ProductID',
      'ChannelID',
      'LocationID',
      'Period',
      'ADI',
      'CV2',
      'Category',
      'Algorithm',
      'IsActive',
      'CreatedAt'
    ];

    const lines = this.rows.map(r =>
      [
        r.scenario_id ?? this.currentScenarioId,
        r.ProductID,
        r.ChannelID,
        r.LocationID,
        r.Period,
        r.adi ?? '',
        r.cv2 ?? '',
        r.category,
        r.algorithm,
        r.isActive ?? '',
        r.created_at ?? ''
      ]
        .map(v => `"${String(v).replace(/"/g, '""')}"`)
        .join(',')
    );

    const csv = [header.join(','), ...lines].join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download =
      `Classified_Results_s${this.currentScenarioId}_${this.period.value}` +
      `${this.includeInactive.value ? '_with_inactive' : ''}.csv`;

    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  }

  runInPlanning(): void {
    this.router.navigate(['/planning-run/classify-forecast-elements'], {
      queryParams: {
        period: this.period.value
      }
    });
  }
}