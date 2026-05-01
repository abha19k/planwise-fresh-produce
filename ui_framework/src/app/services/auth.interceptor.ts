import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from './auth.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getAccessToken();

  if (req.url.includes('/api/login') || req.url.includes('/api/refresh')) {
    return next(req);
  }

  let url = req.url;

  if (url.includes('127.0.0.1:8000') && !url.includes('db_schema=')) {
    const separator = url.includes('?') ? '&' : '?';
    url = `${url}${separator}db_schema=planwise_fresh_produce`;
  }

  const modifiedReq = req.clone({
    url,
    setHeaders: token
      ? {
          Authorization: `Bearer ${token}`,
        }
      : {},
  });

  return next(modifiedReq);
};