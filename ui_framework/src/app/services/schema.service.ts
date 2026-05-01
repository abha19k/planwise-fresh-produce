import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class SchemaService {
  private key = 'planwise_schema';

  schema = signal<string>(this.loadSchema());

  setSchema(s: string) {
    localStorage.setItem(this.key, s);
    this.schema.set(s);
  }

  getSchema(): string {
    return this.schema();
  }

  private loadSchema(): string {
    return localStorage.getItem(this.key) || 'planwise_fresh_produce';
  }
}