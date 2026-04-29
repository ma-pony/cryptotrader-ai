import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        success: {
          DEFAULT: 'hsl(var(--success))',
          foreground: 'hsl(var(--success-foreground))',
        },
        warning: {
          DEFAULT: 'hsl(var(--warning))',
          foreground: 'hsl(var(--warning-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        amber: {
          50: 'var(--amber-50)',
          200: 'var(--amber-200)',
          400: 'var(--amber-400)',
          500: 'var(--amber-500)',
          600: 'var(--amber-600)',
        },
        cyan: {
          400: 'var(--cyan-400)',
          500: 'var(--cyan-500)',
          600: 'var(--cyan-600)',
        },
        violet: {
          400: 'var(--violet-400)',
          500: 'var(--violet-500)',
          600: 'var(--violet-600)',
        },
        trade: {
          long: 'var(--trade-long)',
          'long-soft': 'var(--trade-long-soft)',
          short: 'var(--trade-short)',
          'short-soft': 'var(--trade-short-soft)',
        },
        agent: {
          tech: 'var(--agent-tech)',
          chain: 'var(--agent-chain)',
          news: 'var(--agent-news)',
          macro: 'var(--agent-macro)',
          verdict: 'var(--agent-verdict)',
        },
      },
      boxShadow: {
        'glow-amber': '0 0 32px var(--amber-glow)',
        'glow-violet': '0 0 32px var(--violet-glow)',
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
