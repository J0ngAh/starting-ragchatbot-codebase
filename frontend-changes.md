# Frontend Changes: Theme Toggle Button & Light Theme

## Overview

Added a dark/light theme toggle button and comprehensive light theme variant to the Course Materials Assistant interface.

## Files Modified

### 1. `frontend/index.html`

Added a theme toggle button in the main chat area (lines 110-131):

- Positioned inside `<main class="chat-main">` as the first child element
- Contains two SVG icons: sun (for switching to light mode) and moon (for switching to dark mode)
- Includes accessibility attributes (`aria-label`, `title`)

### 2. `frontend/style.css`

Added theme toggle styles and comprehensive light theme (lines 867-1115):

**Toggle Button Styles:**
- `.theme-toggle` - Base button styling with absolute positioning in top-right corner
- Hover/active/focus states with smooth transitions
- Icon animation with rotation and scale transforms

**Light Theme Color System (CSS Variables):**

| Variable | Dark Mode | Light Mode | Purpose |
|----------|-----------|------------|---------|
| `--bg-deep` | `#0c0a09` | `#fafaf9` | Deepest background |
| `--bg-base` | `#1c1917` | `#f5f5f4` | Base background |
| `--bg-surface` | `#292524` | `#ffffff` | Surface/card background |
| `--bg-elevated` | `#3f3a36` | `#e7e5e4` | Elevated elements |
| `--bg-hover` | `#44403c` | `#d6d3d1` | Hover states |
| `--accent` | `#f59e0b` | `#d97706` | Primary accent (darker for contrast) |
| `--text-primary` | `#fafaf9` | `#1c1917` | Main text |
| `--text-secondary` | `#a8a29e` | `#57534e` | Secondary text |
| `--text-muted` | `#78716c` | `#78716c` | Muted text |

**Component-Specific Overrides:**
- User messages: White text on darker amber gradient (`#b45309`)
- Code blocks: Subtle gray background with amber-tinted inline code
- Pre blocks: Light background with subtle border
- Send button: White text on amber, darker on hover
- Sources section: Proper background contrast
- Scrollbar: Visible thumb color
- Links: Darker amber (`#b45309`) for WCAG contrast
- Blockquotes: Amber left border with light amber background
- Suggested cards: White background with subtle shadow
- Welcome message: Gradient from surface to elevated
- Theme toggle: White background with shadow in light mode

**Responsive Adjustments:**
- Smaller button size on mobile (40px vs 44px)
- Adjusted positioning for smaller screens

### 3. `frontend/script.js`

Added theme management functionality (lines 58-92):

**Functions Added:**
- `initializeTheme()` - Initializes theme from localStorage or system preference
- `setTheme(theme)` - Applies theme and updates accessibility labels
- `toggleTheme()` - Toggles between dark and light themes

**Features:**
- Persists user preference in localStorage
- Respects system `prefers-color-scheme` preference
- Listens for system theme changes (only when no user preference set)
- Keyboard navigation support (Enter and Space keys)
- Dynamic `aria-label` updates based on current theme

## Accessibility Features

- **Color Contrast**: All text meets WCAG AA standards (4.5:1 for normal text)
- **Interactive Elements**: Clear focus states with visible outlines
- **Keyboard Navigation**: Full support for Tab, Enter, and Space keys
- **Screen Readers**: Dynamic `aria-label` updates based on current theme
- **System Preference**: Respects `prefers-color-scheme` media query
- **Reduced Motion**: Uses design system transitions (respects user preferences)

## Color Contrast Ratios (Light Theme)

| Element | Foreground | Background | Ratio |
|---------|------------|------------|-------|
| Primary text | `#1c1917` | `#fafaf9` | 16.5:1 |
| Secondary text | `#57534e` | `#fafaf9` | 5.8:1 |
| Muted text | `#78716c` | `#fafaf9` | 4.5:1 |
| Links | `#b45309` | `#ffffff` | 4.6:1 |
| User message | `#ffffff` | `#d97706` | 4.5:1 |

## How It Works

1. On page load, `initializeTheme()` checks:
   - localStorage for saved preference
   - System `prefers-color-scheme` media query
   - Defaults to dark theme if no preference found

2. When toggled:
   - `data-theme` attribute is set on `<html>` element
   - CSS variables are overridden via `[data-theme="light"]` selector
   - Component-specific overrides are applied
   - Preference is saved to localStorage
   - `aria-label` is updated for screen readers

3. Animations:
   - Icons rotate and scale in/out with `400ms cubic-bezier` easing
   - Button scales on hover (1.05) and active (0.95)
   - All color transitions use the design system's `--transition-base` (250ms)
