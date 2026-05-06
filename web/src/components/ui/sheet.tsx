import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { forwardRef, type ComponentPropsWithoutRef, type ElementRef } from 'react';

import { cn } from '@/lib/cn';

/**
 * Sheet — edge-anchored modal panel built on @radix-ui/react-dialog. Used for
 * mobile sidebar drawers and chat session list. Functionally identical to
 * Dialog but the content is pinned to a screen edge instead of centered, so
 * it works as a "slide in from the side" drawer pattern.
 *
 * Animations are intentionally bare (no tailwindcss-animate dependency) —
 * Radix still handles the open/close state cleanly; if motion is wanted
 * later, install the plugin and add ``data-[state=open]:slide-in-from-*``
 * utilities to ``sideClasses``.
 */
export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;
export const SheetPortal = DialogPrimitive.Portal;

const SheetOverlay = forwardRef<
  ElementRef<typeof DialogPrimitive.Overlay>,
  ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm',
      className,
    )}
    {...props}
  />
));
SheetOverlay.displayName = 'SheetOverlay';

const sideClasses: Record<'left' | 'right' | 'top' | 'bottom', string> = {
  left: 'inset-y-0 left-0 h-full w-[min(20rem,85vw)] border-r border-border',
  right: 'inset-y-0 right-0 h-full w-[min(20rem,85vw)] border-l border-border',
  top: 'inset-x-0 top-0 max-h-[85vh] border-b border-border',
  bottom: 'inset-x-0 bottom-0 max-h-[85vh] border-t border-border',
};

export interface SheetContentProps
  extends ComponentPropsWithoutRef<typeof DialogPrimitive.Content> {
  side?: 'left' | 'right' | 'top' | 'bottom';
  /** Hide the default close (X) affordance. */
  hideClose?: boolean;
}

export const SheetContent = forwardRef<
  ElementRef<typeof DialogPrimitive.Content>,
  SheetContentProps
>(({ side = 'left', className, children, hideClose = false, ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed z-50 flex flex-col bg-card shadow-xl focus:outline-none',
        sideClasses[side],
        className,
      )}
      {...props}
    >
      {children}
      {hideClose ? null : (
        <DialogPrimitive.Close
          className="absolute right-3 top-3 rounded-sm text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </DialogPrimitive.Close>
      )}
    </DialogPrimitive.Content>
  </SheetPortal>
));
SheetContent.displayName = 'SheetContent';

export const SheetTitle = forwardRef<
  ElementRef<typeof DialogPrimitive.Title>,
  ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn('text-lg font-semibold', className)}
    {...props}
  />
));
SheetTitle.displayName = 'SheetTitle';
