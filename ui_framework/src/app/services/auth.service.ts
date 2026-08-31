import { Injectable, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { tap } from 'rxjs';

export type UserRole = 'admin' | 'planner' | 'viewer';

export interface AuthUser {
  id: string;
  email: string;
  full_name?: string;
  role: UserRole;
  roles: UserRole[];
  is_active: boolean;
}

export interface LoginResponse {
  user: AuthUser;
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private apiUrl = 'http://127.0.0.1:8000';

  private accessTokenKey = 'planwise_access_token';
  private refreshTokenKey = 'planwise_refresh_token';
  private userKey = 'planwise_user';

  user = signal<AuthUser | null>(this.loadUser());

  isLoggedIn(): boolean {
    return this.hasValidAccessToken();
  }
  role = computed(() => this.user()?.role ?? null);
  roles = computed(() => this.user()?.roles ?? []);

  constructor(
    private http: HttpClient,
    private router: Router,
  ) {}

  login(email: string, password: string) {
    return this.http.post<LoginResponse>(`${this.apiUrl}/api/login`, {
      email,
      password,
    }).pipe(
      tap((res) => {
        localStorage.setItem(this.accessTokenKey, res.access_token);
        localStorage.setItem(this.refreshTokenKey, res.refresh_token);
        localStorage.setItem(this.userKey, JSON.stringify(res.user));
        this.user.set(res.user);
      })
    );
  }

  logout() {
    const refreshToken = this.getRefreshToken();

    if (refreshToken) {
      this.http.post(`${this.apiUrl}/api/logout`, {
        refresh_token: refreshToken,
      }).subscribe({
        next: () => this.clearSession(),
        error: () => this.clearSession(),
      });
    } else {
      this.clearSession();
    }
  }

  clearSession() {
    localStorage.removeItem(this.accessTokenKey);
    localStorage.removeItem(this.refreshTokenKey);
    localStorage.removeItem(this.userKey);
    this.user.set(null);
    this.router.navigate(['/login']);
  }

  getAccessToken(): string | null {
    return localStorage.getItem(this.accessTokenKey);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(this.refreshTokenKey);
  }

  hasAnyRole(allowedRoles: UserRole[]): boolean {
    const currentRoles = this.roles();
    return allowedRoles.some(role => currentRoles.includes(role));
  }

  private hasValidAccessToken(): boolean {
    const token = this.getAccessToken();
  
    if (!token) {
      return false;
    }
  
    try {
      const parts = token.split('.');
  
      if (parts.length !== 3) {
        this.clearStoredSession();
        return false;
      }
  
      // JWT uses Base64URL rather than normal Base64
      let base64 = parts[1]
        .replace(/-/g, '+')
        .replace(/_/g, '/');
  
      while (base64.length % 4) {
        base64 += '=';
      }
  
      const payload = JSON.parse(atob(base64));
  
      // No expiry in token
      if (!payload.exp) {
        return true;
      }
  
      const expiresAt = payload.exp * 1000;
  
      if (Date.now() >= expiresAt) {
        this.clearStoredSession();
        return false;
      }
  
      return true;
  
    } catch {
      this.clearStoredSession();
      return false;
    }
  }

  private clearStoredSession(): void {
    localStorage.removeItem(this.accessTokenKey);
    localStorage.removeItem(this.refreshTokenKey);
    localStorage.removeItem(this.userKey);
    this.user.set(null);
  }

  private loadUser(): AuthUser | null {
    const raw = localStorage.getItem(this.userKey);
    if (!raw) return null;

    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
}