import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import {
  FormBuilder,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators
} from '@angular/forms';
import { HttpClient, HttpClientModule, HttpParams } from '@angular/common/http';
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
import { firstValueFrom } from 'rxjs';

import { ScenarioService } from '../../../services/scenario.service';

/** --- Backend contracts --- */
interface KeyTriplet {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
}

interface SearchResult {
  query: string;
  count: number;
  keys: KeyTriplet[];
}

interface SavedSearch {
  id?: number;
  name: string;
  query: string;
  created_at?: string;
}

interface ClassificationRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  periods: number;
  nonzero_count: number;
  adi: number | null;
  cv2: number | null;
  category: 'Smooth' | 'Intermittent' | 'Erratic' | 'Sparse' | 'NotEnoughHistory';
  seasonal?: boolean | null;
  createdAt?: string;
}

interface HistoryRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  StartDate: string;
  Qty: number;
}

interface BackendClassRow {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
  Period: string;
  Label: string;
  Score: number;
  IsActive: boolean;
  ComputedAt: string;
  scenario_id?: number;
}

type PeriodView = 'Daily' | 'Weekly' | 'Monthly';
type Algo = 'HoltWinters' | 'XGBoost' | 'MovingAverage' | 'Croston' | 'ARIMA' | 'ETS';

@Component({
  standalone: true,
  selector: 'app-classify-forecast-elements',
  templateUrl: './classify-forecast-elements.component.html',
  styleUrls: ['./classify-forecast-elements.component.scss'],
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
export class ClassifyForecastElementsComponent implements OnInit {
  private http = inject(HttpClient);
  private fb = inject(FormBuilder);

  readonly scenarioService = inject(ScenarioService);

  readonly API = 'http://127.0.0.1:8000/api';
  readonly DB_SCHEMA = 'planwise_fresh_produce';

  savedSearches: SavedSearch[] = [];
  selectedSavedIndex = new FormControl<number>(-1, { nonNullable: true });
  periodSelection = new FormControl<PeriodView>('Daily', { nonNullable: true });

  loading = false;
  errorMsg: string | null = null;

  saving = false;
  saveMsg: string | null = null;
  saveErr: string | null = null;

  rows: ClassificationRow[] = [];

  form: FormGroup = this.fb.group({
    autoEnabled: this.fb.control<boolean>(true, { nonNullable: true }),
    manualEnabled: this.fb.control<boolean>(false, { nonNullable: true }),

    autoLookback: this.fb.control<number>(12, {
      nonNullable: true,
      validators: [Validators.min(1)]
    }),
    autoAlgo: this.fb.control<Algo>('XGBoost', { nonNullable: true }),

    manual: this.fb.group({
      Smooth: this.fb.control<Algo>('HoltWinters', { nonNullable: true }),
      Intermittent: this.fb.control<Algo>('XGBoost', { nonNullable: true }),
      Erratic: this.fb.control<Algo>('XGBoost', { nonNullable: true }),
      Sparse: this.fb.control<Algo>('MovingAverage', { nonNullable: true }),
      NotEnoughHistory: this.fb.control<Algo>('XGBoost', { nonNullable: true })
    })
  });

  readonly algoOptions: Algo[] = [
    'XGBoost',
    'HoltWinters',
    'MovingAverage',
    'Croston',
    'ARIMA',
    'ETS'
  ];

  get autoEnabledCtrl(): FormControl<boolean> {
    return this.form.get('autoEnabled') as FormControl<boolean>;
  }

  get manualEnabledCtrl(): FormControl<boolean> {
    return this.form.get('manualEnabled') as FormControl<boolean>;
  }

  get autoLookbackCtrl(): FormControl<number> {
    return this.form.get('autoLookback') as FormControl<number>;
  }

  get autoAlgoCtrl(): FormControl<Algo> {
    return this.form.get('autoAlgo') as FormControl<Algo>;
  }

  get manualGroup(): FormGroup {
    return this.form.get('manual') as FormGroup;
  }

  ngOnInit(): void {
    this.refreshSavedSearches();

    this.autoEnabledCtrl.valueChanges.subscribe(v => {
      if (v) {
        this.manualEnabledCtrl.setValue(false, { emitEvent: false });
      } else if (!this.manualEnabledCtrl.value) {
        this.manualEnabledCtrl.setValue(true, { emitEvent: false });
      }
    });

    this.manualEnabledCtrl.valueChanges.subscribe(v => {
      if (v) {
        this.autoEnabledCtrl.setValue(false, { emitEvent: false });
      } else if (!this.autoEnabledCtrl.value) {
        this.autoEnabledCtrl.setValue(true, { emitEvent: false });
      }
    });
  }

  get currentScenarioId(): number {
    return this.scenarioService.selectedScenarioId();
  }

  private schemaParams(): HttpParams {
    return new HttpParams().set('db_schema', this.DB_SCHEMA);
  }

  private baseParams(): HttpParams {
    return new HttpParams()
      .set('db_schema', this.DB_SCHEMA)
      .set('scenario_id', this.currentScenarioId);
  }

  refreshSavedSearches(): void {
    this.http.get<SavedSearch[]>(`${this.API}/saved-searches`, {
      params: this.schemaParams()
    }).subscribe({
      next: rows => {
        this.savedSearches = rows || [];
      },
      error: () => {
        this.savedSearches = [];
      }
    });
  }

  private periodSlug(): 'daily' | 'weekly' | 'monthly' {
    const p = this.periodSelection.value;
    return p === 'Weekly' ? 'weekly' : p === 'Monthly' ? 'monthly' : 'daily';
  }

  chosenAlgo(category: ClassificationRow['category']): string {
    if (this.manualEnabledCtrl.value) {
      const manualValues = this.manualGroup.value as Record<string, Algo>;
      return manualValues[category] ?? '(none)';
    }
    return this.autoAlgoCtrl.value;
  }

  trackByKey = (_: number, r: ClassificationRow): string =>
    `${r.ProductID}||${r.ChannelID}||${r.LocationID}`;

  async runClassification(): Promise<void> {
    this.errorMsg = null;
    this.saveMsg = null;
    this.saveErr = null;
    this.rows = [];

    const idx = this.selectedSavedIndex.value;
    if (idx < 0 || idx >= this.savedSearches.length) {
      this.errorMsg = 'Please select a saved search.';
      return;
    }

    const saved = this.savedSearches[idx];
    const q = (saved.query || '').trim();
    if (!q) {
      this.errorMsg = 'Saved search query is empty.';
      return;
    }

    this.loading = true;

    try {
      const searchParams = this.schemaParams()
        .set('q', q)
        .set('limit', 20000)
        .set('offset', 0);

      const sr = await firstValueFrom(
        this.http.get<SearchResult>(`${this.API}/search`, { params: searchParams })
      );

      const keys = sr?.keys ?? [];
      if (!keys.length) {
        this.errorMsg = 'No matches for this saved query.';
        return;
      }

      const computePayload = {
        period: this.periodSlug(),
        scenario_id: this.currentScenarioId,
        lookback_buckets: Number(this.autoLookbackCtrl.value ?? 12),
        min_sum: 1.0
      };

      await firstValueFrom(
        this.http.post(`${this.API}/classify/compute`, computePayload, {
          params: this.schemaParams()
        })
      );

      const resParams = this.baseParams()
        .set('period', this.periodSlug())
        .set('include_inactive', 'true');

      const allRes = await firstValueFrom(
        this.http.get<BackendClassRow[]>(`${this.API}/classify/results`, { params: resParams })
      );

      if (!allRes?.length) {
        this.errorMsg = 'No classification results found for this scenario. Please run Cleanse History first.';
        return;
      }

      const keySet = new Set(keys.map(k => `${k.ProductID}||${k.ChannelID}||${k.LocationID}`));
      const filtered = (allRes || []).filter(r =>
        keySet.has(`${r.ProductID}||${r.ChannelID}||${r.LocationID}`)
      );

      if (!filtered.length) {
        this.errorMsg = 'No scenario cleansed history found for these keys. Please run Cleanse History first.';
        return;
      }

      const histRows = await firstValueFrom(
        this.http.post<HistoryRow[]>(
          `${this.API}/history/${this.periodSlug()}-by-keys`,
          { keys },
          { params: this.schemaParams() }
        )
      );

      const seriesMap = new Map<string, number[]>();
      for (const h of histRows || []) {
        const key = `${h.ProductID}||${h.ChannelID}||${h.LocationID}`;
        const qty = Number(h.Qty) || 0;

        if (!seriesMap.has(key)) {
          seriesMap.set(key, []);
        }
        seriesMap.get(key)!.push(qty);
      }

      const minNonZero = 6;

      const computeMetrics = (qtys: number[]): {
        periods: number;
        nonzero: number;
        adi: number | null;
        cv2: number | null;
        category: ClassificationRow['category'];
      } => {
        const periods = qtys.length;
        const nonzeroVals = qtys.filter(q => q > 0);
        const nonzero = nonzeroVals.length;

        if (!periods || nonzero < minNonZero) {
          return { periods, nonzero, adi: null, cv2: null, category: 'NotEnoughHistory' };
        }

        const adi = periods / nonzero;
        const mean = nonzeroVals.reduce((s, q) => s + q, 0) / nonzeroVals.length;

        if (!(mean > 0) || nonzeroVals.length < 2) {
          return { periods, nonzero, adi, cv2: null, category: 'NotEnoughHistory' };
        }

        const variance =
          nonzeroVals.reduce((s, q) => s + (q - mean) * (q - mean), 0) /
          (nonzeroVals.length - 1);

        const std = Math.sqrt(variance);
        const cv2 = (std / mean) * (std / mean);

        let category: ClassificationRow['category'] = 'Sparse';
        if (adi < 1.32 && cv2 < 0.49) category = 'Smooth';
        else if (adi < 1.32 && cv2 >= 0.49) category = 'Erratic';
        else if (adi >= 1.32 && cv2 < 0.49) category = 'Intermittent';
        else category = 'Sparse';

        return { periods, nonzero, adi, cv2, category };
      };

      this.rows = filtered.map(r => {
        const key = `${r.ProductID}||${r.ChannelID}||${r.LocationID}`;
        const qtys = seriesMap.get(key) || [];
        const m = computeMetrics(qtys);

        return {
          ProductID: r.ProductID,
          ChannelID: r.ChannelID,
          LocationID: r.LocationID,
          periods: m.periods,
          nonzero_count: m.nonzero,
          adi: m.adi,
          cv2: m.cv2,
          category: m.category,
          seasonal: null,
          createdAt: r.ComputedAt
        };
      });

      if (!this.rows.length) {
        this.errorMsg = 'Run succeeded but produced 0 rows after filtering.';
      }
    } catch (e: any) {
      this.errorMsg = e?.error?.detail || e?.message || 'Failed to classify.';
    } finally {
      this.loading = false;
    }
  }

  async saveResults(): Promise<void> {
    this.saveMsg = null;
    this.saveErr = null;

    if (!this.rows.length) {
      this.saveErr = 'Nothing to save. Run classification first.';
      return;
    }

    const payload = {
      period: this.periodSlug(),
      scenario_id: this.currentScenarioId,
      rows: this.rows.map(r => ({
        ProductID: r.ProductID,
        ChannelID: r.ChannelID,
        LocationID: r.LocationID,
        ADI: r.adi,
        CV2: r.cv2,
        Category: r.category,
        Algorithm: this.chosenAlgo(r.category),
        CreatedAt: r.createdAt ?? null
      }))
    };

    this.saving = true;
    try {
      const res = await firstValueFrom(
        this.http.post<{ ok: boolean; count: number; scenario_id?: number }>(
          `${this.API}/classify/save`,
          payload,
          { params: this.schemaParams() }
        )
      );
      this.saveMsg = `Saved ${res?.count ?? payload.rows.length} result(s) for scenario ${this.currentScenarioId}.`;
    } catch (e: any) {
      this.saveErr = e?.error?.detail || e?.message || 'Failed to save results.';
    } finally {
      this.saving = false;
    }
  }
}