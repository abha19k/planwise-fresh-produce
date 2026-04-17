import { Injectable, computed, signal } from '@angular/core';

export interface Scenario {
  scenario_id: number;
  name: string;
  parent_scenario_id?: number | null;
  is_base: boolean;
  created_at?: string | null;
  created_by?: string | null;
  status: string;
}

@Injectable({
  providedIn: 'root'
})
export class ScenarioService {
  private readonly STORAGE_KEY = 'planwise_selected_scenario_id';

  private _scenarios = signal<Scenario[]>([]);
  private _selectedScenarioId = signal<number>(1);

  readonly scenarios = computed(() => this._scenarios());
  readonly selectedScenarioId = computed(() => this._selectedScenarioId());
  readonly selectedScenario = computed(() =>
    this._scenarios().find(s => s.scenario_id === this._selectedScenarioId()) ?? null
  );

  constructor() {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(this.STORAGE_KEY);
      if (saved) {
        const n = Number(saved);
        if (!Number.isNaN(n) && n > 0) {
          this._selectedScenarioId.set(n);
        }
      }
    }
  }

  setScenarios(items: Scenario[]): void {
    const scenarios = items ?? [];
    this._scenarios.set(scenarios);

    const current = this._selectedScenarioId();
    const exists = scenarios.some(x => x.scenario_id === current);

    if (!exists && scenarios.length > 0) {
      const base = scenarios.find(x => x.is_base) ?? scenarios[0];
      this.setSelectedScenarioId(base.scenario_id);
    }
  }

  setSelectedScenarioId(id: number): void {
    this._selectedScenarioId.set(id);

    if (typeof window !== 'undefined') {
      localStorage.setItem(this.STORAGE_KEY, String(id));
    }
  }

  clear(): void {
    this._scenarios.set([]);
    this._selectedScenarioId.set(1);

    if (typeof window !== 'undefined') {
      localStorage.removeItem(this.STORAGE_KEY);
    }
  }
}