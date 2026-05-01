import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';

import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  ColComponent,
  ContainerComponent,
  FormControlDirective,
  InputGroupComponent,
  InputGroupTextDirective,
  RowComponent
} from '@coreui/angular';

import { IconDirective } from '@coreui/icons-angular';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ContainerComponent,
    RowComponent,
    ColComponent,
    CardComponent,
    CardBodyComponent,
    InputGroupComponent,
    InputGroupTextDirective,
    FormControlDirective,
    ButtonDirective,
    IconDirective
  ],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  email = '';
  password = '';
  loading = false;
  errorMessage = '';

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}
  
  login(): void {
    console.log('Login clicked'); // ADD THIS
  
    this.errorMessage = '';
  
    if (!this.email || !this.password) {
      this.errorMessage = 'Please enter email and password.';
      return;
    }
  
    this.loading = true;
  
    this.authService.login(this.email, this.password).subscribe({
      next: (res) => {
        console.log('Login success', res); // ADD THIS
  
        this.loading = false;
  
        this.router.navigate(['/dashboard']); // SHOULD RUN
      },
      error: (err) => {
        console.log('Login error', err); // ADD THIS
  
        this.loading = false;
        this.errorMessage = err?.error?.detail || 'Login failed.';
      }
    });
  }
}
