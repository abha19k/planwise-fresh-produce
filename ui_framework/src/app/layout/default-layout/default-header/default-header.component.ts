import { NgTemplateOutlet } from '@angular/common';
import { Component, computed, inject, input } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

import {
  AvatarComponent,
  BreadcrumbRouterComponent,
  ColorModeService,
  ContainerComponent,
  DropdownComponent,
  DropdownDividerDirective,
  DropdownHeaderDirective,
  DropdownItemDirective,
  DropdownMenuDirective,
  DropdownToggleDirective,
  HeaderComponent,
  HeaderNavComponent,
  HeaderTogglerDirective,
  NavItemComponent,
  NavLinkDirective,
  SidebarToggleDirective
} from '@coreui/angular';

import { IconDirective } from '@coreui/icons-angular';
import { AuthService } from '../../../services/auth.service';

@Component({
  selector: 'app-default-header',
  templateUrl: './default-header.component.html',
  imports: [
    ContainerComponent,
    HeaderTogglerDirective,
    SidebarToggleDirective,
    IconDirective,
    HeaderNavComponent,
    NavItemComponent,
    NavLinkDirective,
    RouterLink,
    RouterLinkActive,
    NgTemplateOutlet,
    BreadcrumbRouterComponent,
    DropdownComponent,
    DropdownToggleDirective,
    AvatarComponent,
    DropdownMenuDirective,
    DropdownHeaderDirective,
    DropdownItemDirective,
    DropdownDividerDirective
  ]
})
export class DefaultHeaderComponent extends HeaderComponent {

  // ✅ Services
  readonly #colorModeService = inject(ColorModeService);
  readonly authService = inject(AuthService);

  // ✅ Reactive values
  readonly colorMode = this.#colorModeService.colorMode;
  readonly user = this.authService.user;

  // ✅ Sidebar
  sidebarId = input('sidebar1');

  // ✅ Theme options
  readonly colorModes = [
    { name: 'light', text: 'Light', icon: 'cilSun' },
    { name: 'dark', text: 'Dark', icon: 'cilMoon' },
    { name: 'auto', text: 'Auto', icon: 'cilContrast' }
  ];

  readonly icons = computed(() => {
    const currentMode = this.colorMode();
    return this.colorModes.find(mode => mode.name === currentMode)?.icon ?? 'cilSun';
  });

  // --------------------------------------------------
  // ✅ USER DISPLAY HELPERS
  // --------------------------------------------------

  userDisplayName(): string {
    const u = this.user() as any;
    return u?.full_name || u?.email || 'User';
  }

  userEmail(): string {
    const u = this.user() as any;
    return u?.email || '';
  }

  userRole(): string {
    const u = this.user() as any;
    return u?.role || '';
  }

  userInitials(): string {
    const u = this.user() as any;
    const value = u?.full_name || u?.email || 'U';

    return value
      .split(/[ .@_-]+/)
      .filter((part: string) => !!part)
      .map((part: string) => part.charAt(0))
      .join('')
      .slice(0, 2)
      .toUpperCase();
  }

  // --------------------------------------------------
  // ✅ AVATAR HANDLING
  // --------------------------------------------------

  avatarUrl(): string {
    const u = this.user() as any;
    const file = u?.profile_image_url;

    // If user has image → load from assets
    if (file) {
      return `assets/images/avatars/${file}`;
    }

    // fallback image
    return `assets/images/avatars/${file}`;
  }

  hasAvatar(): boolean {
    const u = this.user() as any;
    return !!u?.profile_image_url;
  }

  // --------------------------------------------------
  // ✅ LOGOUT
  // --------------------------------------------------

  logout(): void {
    this.authService.logout();
  }
}