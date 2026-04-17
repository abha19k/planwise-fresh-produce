import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  CardHeaderComponent,
  ColComponent,
  RowComponent,
  TableDirective
} from '@coreui/angular';

import { ScenarioApiService } from '../../../services/scenario-api.service';
import { ScenarioService, Scenario } from '../../../services/scenario.service';

@Component({
  selector: 'app-scenario-manager',
  standalone: true,
  templateUrl: './scenario-manager.component.html',
  styleUrls: ['./scenario-manager.component.scss'],
  imports: [
    CommonModule,
    FormsModule,
    CardComponent,
    CardHeaderComponent,
    CardBodyComponent,
    RowComponent,
    ColComponent,
    TableDirective,
    ButtonDirective
  ]
})
export class ScenarioManagerComponent implements OnInit {
  private scenarioApi = inject(ScenarioApiService);
  readonly scenarioService = inject(ScenarioService);

  dbSchema = 'planwise_fresh_produce';

  loading = false;
  copying = false;
  error = '';

  newScenarioName = '';
  createdBy = 'abha';

  ngOnInit(): void {
    this.loadScenarios();
  }

  loadScenarios(): void {
    this.loading = true;
    this.error = '';

    this.scenarioApi.getScenarios(this.dbSchema).subscribe({
      next: (items) => {
        this.scenarioService.setScenarios(items ?? []);
        this.loading = false;
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Failed to load scenarios';
        this.loading = false;
      }
    });
  }

  selectScenario(id: string | number): void {
    const scenarioId = Number(id);
    if (!Number.isNaN(scenarioId) && scenarioId > 0) {
      this.scenarioService.setSelectedScenarioId(scenarioId);
    }
  }

  copyScenario(): void {
    this.error = '';

    const parentScenarioId = this.scenarioService.selectedScenarioId();
    const name = this.newScenarioName.trim();

    if (!name) {
      this.error = 'Please enter a scenario name.';
      return;
    }

    this.copying = true;

    this.scenarioApi.copyScenario(parentScenarioId, this.dbSchema, name, this.createdBy).subscribe({
      next: (created) => {
        this.newScenarioName = '';
        this.copying = false;

        this.loadScenarios();
        this.scenarioService.setSelectedScenarioId(created.scenario_id);
      },
      error: (err) => {
        this.error = err?.error?.detail || 'Failed to copy scenario';
        this.copying = false;
      }
    });
  }

  get scenarios(): Scenario[] {
    return this.scenarioService.scenarios();
  }

  isSelectedScenario(id: number): boolean {
    return this.scenarioService.selectedScenarioId() === id;
  }

  parentScenarioName(parentId: number | null | undefined): string {
    if (parentId == null) return '-';
    const parent = this.scenarios.find(s => s.scenario_id === parentId);
    return parent?.name || String(parentId);
  }
}