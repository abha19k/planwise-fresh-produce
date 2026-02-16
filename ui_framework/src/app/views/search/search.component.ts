import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, signal, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

type Domain = 'product' | 'channel' | 'location';
type AndOr = 'AND' | 'OR';

interface FieldDef {
  domain: Domain;
  key: string;   // must match backend ALLOWED_FIELDS keys (lowercase)
  label: string; // UI label
}

interface Criterion {
  key: string;               // e.g. 'productid'
  value: string;             // e.g. 'AztecWrap' or 'All'
  op?: AndOr;                // AND/OR (undefined for first)
  orValues?: string[];       // when value === 'All', concrete list to expand into OR
}

interface SavedSearchRow {
  id: number;
  name: string;
  query: string;
  created_at: string;
}

interface SearchResultKey {
  ProductID: string;
  ChannelID: string;
  LocationID: string;
}

@Component({
  standalone: true,
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.scss'],
  imports: [CommonModule, FormsModule, HttpClientModule],
})
export class SearchComponent implements OnInit {
  readonly API = 'http://127.0.0.1:8000/api';
  private http = inject(HttpClient);

  // -------------------------
  // BUILDER (Create Saved Search)
  // -------------------------
  fields: FieldDef[] = [
    // product
    { domain: 'product',  key: 'productid',               label: 'ProductID' },
    { domain: 'product',  key: 'productdescr',            label: 'ProductDescr' },
    { domain: 'product',  key: 'businessunit',            label: 'BusinessUnit' },
    { domain: 'product',  key: 'isdailyforecastrequired', label: 'IsDailyForecastRequired' },
    { domain: 'product',  key: 'isnew',                   label: 'IsNew' },
    { domain: 'product',  key: 'productfamily',           label: 'ProductFamily' },
    { domain: 'product',  key: 'productlevel',            label: 'ProductLevel' },

    // channel
    { domain: 'channel',  key: 'channelid',               label: 'ChannelID' },
    { domain: 'channel',  key: 'channeldescr',            label: 'ChannelDescr' },
    { domain: 'channel',  key: 'channellevel',            label: 'ChannelLevel' },

    // location
    { domain: 'location', key: 'locationid',              label: 'LocationID' },
    { domain: 'location', key: 'locationdescr',           label: 'LocationDescr' },
    { domain: 'location', key: 'locationlevel',           label: 'LocationLevel' },
    { domain: 'location', key: 'isactive',                label: 'IsActive' },
  ];

  domain = signal<Domain>('product');
  fieldKey = signal<string>('productid');
  selectedValue = signal<string>('All');
  nextOp = signal<AndOr>('AND');
  criteria = signal<Criterion[]>([]);
  saveName = signal<string>('');

  private currentValues = signal<string[]>([]);
  private optionsCache = new Map<string, string[]>();

  saving = signal<boolean>(false);
  savedOk = signal<boolean>(false);
  error = signal<string | null>(null);

  fieldValues = computed(() => this.currentValues());

  // -------------------------
  // SAVED SEARCHES LIST (view/run/delete)
  // -------------------------
  rows = signal<SavedSearchRow[]>([]);
  loading = signal(false);

  filter = signal<string>('');
  filteredRows = computed(() => {
    const f = (this.filter() || '').trim().toLowerCase();
    if (!f) return this.rows();
    return this.rows().filter(r =>
      r.name.toLowerCase().includes(f) ||
      r.query.toLowerCase().includes(f)
    );
  });

  selectedSaved = signal<SavedSearchRow | null>(null);

  running = signal(false);
  runError = signal<string | null>(null);
  resultCount = signal<number>(0);
  resultKeys = signal<SearchResultKey[]>([]);

  // -------------------------
  // INIT
  // -------------------------
  async ngOnInit() {
    await this.loadOptionsForField(this.fieldKey());
    await this.refreshSaved();
  }

  // -------------------------
  // BUILDER handlers
  // -------------------------
  async onDomainChange(dom: Domain) {
    this.domain.set(dom);
    const firstForDomain = this.fields.find(f => f.domain === dom)?.key || 'productid';
    this.fieldKey.set(firstForDomain);
    await this.loadOptionsForField(firstForDomain);
  }

  async onFieldChange(key: string) {
    this.fieldKey.set(key);
    await this.loadOptionsForField(key);
  }

  async addCriterion() {
    const key = this.fieldKey();
    const value = this.selectedValue();
    const list = await this.ensureOptionsCache(key);

    const crit: Criterion = {
      key,
      value,
      op: this.criteria().length ? this.nextOp() : undefined,
      orValues: value === 'All' ? list.slice() : undefined,
    };

    this.criteria.update(arr => [...arr, crit]);
    this.nextOp.set('AND');
  }

  removeCriterion(i: number) {
    this.criteria.update(arr => arr.filter((_, idx) => idx !== i));
  }

  setOp(i: number, op: AndOr) {
    this.criteria.update(arr => {
      const copy = arr.slice();
      if (i > 0) copy[i] = { ...copy[i], op };
      return copy;
    });
  }

  builtQuery(): string {
    const items = this.criteria();
    if (!items.length) return '';

    const parts: string[] = [];
    items.forEach((c, idx) => {
      const prefix = idx === 0 ? '' : ` ${c.op ?? 'AND'} `;
      parts.push(prefix + this.buildExpr(c));
    });
    return parts.join('');
  }

  private buildExpr(c: Criterion): string {
    const quoteIfNeeded = (v: string) => (/\s/.test(v) ? `"${v}"` : v);

    if (c.value === 'All') {
      const vals = (c.orValues ?? []).filter(v => v && v !== 'All');
      if (!vals.length) return `${c.key}:*`;
      const ors = vals.map(v => `${c.key}:${quoteIfNeeded(v)}`).join(' OR ');
      return `(${ors})`;
    }
    return `${c.key}:${quoteIfNeeded(c.value)}`;
  }

  async save() {
    this.error.set(null);
    this.savedOk.set(false);

    const name = (this.saveName() || '').trim();
    const query = (this.builtQuery() || '').trim();

    if (!name) {
      this.error.set('Please provide a name for this search.');
      return;
    }
    if (!query) {
      this.error.set('Please add at least one criterion.');
      return;
    }

    this.saving.set(true);
    try {
      await firstValueFrom(this.http.post(`${this.API}/saved-searches`, { name, query }));
      this.savedOk.set(true);

      // refresh list and select the newest match
      await this.refreshSaved();
      const latest = this.rows().find(r => r.name === name && r.query === query) || null;
      this.selectedSaved.set(latest);

      // optional: clear builder
      // this.criteria.set([]);
      // this.saveName.set('');
    } catch (e: any) {
      this.error.set(e?.error?.detail || e?.message || 'Failed to save the query.');
    } finally {
      this.saving.set(false);
    }
  }

  private async loadOptionsForField(key: string) {
    const vals = await this.ensureOptionsCache(key);
    const withAll = ['All', ...vals];
    this.currentValues.set(withAll);

    const current = this.selectedValue();
    if (!withAll.includes(current)) {
      this.selectedValue.set('All');
    }
  }

  private async ensureOptionsCache(key: string): Promise<string[]> {
    if (this.optionsCache.has(key)) return this.optionsCache.get(key)!;
    const vals = await this.fetchValuesForKey(key);
    this.optionsCache.set(key, vals);
    return vals;
  }

  private async fetchValuesForKey(key: string): Promise<string[]> {
    try {
      if (key.startsWith('product')) {
        const rows = await firstValueFrom(this.http.get<any[]>(`${this.API}/products`));
        return this.distinctSorted(rows.map(r => this.pickProductField(key, r)));
      }
      if (key.startsWith('channel')) {
        const rows = await firstValueFrom(this.http.get<any[]>(`${this.API}/channels`));
        return this.distinctSorted(rows.map(r => this.pickChannelField(key, r)));
      }
      const rows = await firstValueFrom(this.http.get<any[]>(`${this.API}/locations`));
      return this.distinctSorted(rows.map(r => this.pickLocationField(key, r)));
    } catch {
      return [];
    }
  }

  private pickProductField(key: string, r: any): string {
    switch (key) {
      case 'productid':               return String(r.ProductID ?? '');
      case 'productdescr':            return String(r.ProductDescr ?? '');
      case 'businessunit':            return String(r.BusinessUnit ?? '');
      case 'isdailyforecastrequired': return String(r.IsDailyForecastRequired ?? '');
      case 'isnew':                   return String(r.IsNew ?? '');
      case 'productfamily':           return String(r.ProductFamily ?? '');
      case 'productlevel':            return String(r.Level ?? '');
      default:                        return '';
    }
  }

  private pickChannelField(key: string, r: any): string {
    switch (key) {
      case 'channelid':    return String(r.ChannelID ?? '');
      case 'channeldescr': return String(r.ChannelDescr ?? '');
      case 'channellevel': return String(r.Level ?? '');
      default:             return '';
    }
  }

  private pickLocationField(key: string, r: any): string {
    switch (key) {
      case 'locationid':    return String(r.LocationID ?? '');
      case 'locationdescr': return String(r.LocationDescr ?? '');
      case 'locationlevel': return String(r.Level ?? '');
      case 'isactive':      return String(r.IsActive ?? '');
      default:              return '';
    }
  }

  private distinctSorted(arr: string[]): string[] {
    return Array.from(new Set(arr.filter(Boolean))).sort((a, b) => a.localeCompare(b));
  }

  // -------------------------
  // SAVED SEARCHES handlers
  // -------------------------
  async refreshSaved() {
    this.loading.set(true);
    this.error.set(null);
    try {
      const data = await firstValueFrom(this.http.get<SavedSearchRow[]>(`${this.API}/saved-searches`));
      this.rows.set(data || []);

      const sel = this.selectedSaved();
      if (sel) this.selectedSaved.set((data || []).find(x => x.id === sel.id) || null);
    } catch (e: any) {
      this.error.set(e?.error?.detail || e?.message || 'Failed to load saved searches.');
    } finally {
      this.loading.set(false);
    }
  }

  selectSaved(r: SavedSearchRow) {
    this.selectedSaved.set(r);
    this.runError.set(null);
    this.resultCount.set(0);
    this.resultKeys.set([]);
  }

  async runSelected(limit = 2000, offset = 0) {
    const sel = this.selectedSaved();
    if (!sel) return;

    this.running.set(true);
    this.runError.set(null);
    try {
      const res = await firstValueFrom(
        this.http.get<any>(`${this.API}/search`, {
          params: { q: sel.query, limit: String(limit), offset: String(offset) },
        })
      );
      this.resultCount.set(Number(res?.count || 0));
      this.resultKeys.set((res?.keys || []) as SearchResultKey[]);
    } catch (e: any) {
      this.runError.set(e?.error?.detail || e?.message || 'Failed to run search.');
    } finally {
      this.running.set(false);
    }
  }

  async deleteSelected() {
    const sel = this.selectedSaved();
    if (!sel) return;

    const ok = window.confirm(`Delete saved search "${sel.name}"?`);
    if (!ok) return;

    this.loading.set(true);
    this.error.set(null);
    try {
      await firstValueFrom(this.http.delete(`${this.API}/saved-searches/${sel.id}`));
      this.selectedSaved.set(null);
      await this.refreshSaved();
    } catch (e: any) {
      this.error.set(e?.error?.detail || e?.message || 'Failed to delete saved search.');
    } finally {
      this.loading.set(false);
    }
  }

  async copyQuery() {
    const sel = this.selectedSaved();
    if (!sel) return;
    await navigator.clipboard.writeText(sel.query);
  }
}
