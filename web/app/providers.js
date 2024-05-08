"use client";

import {NextUIProvider} from '@nextui-org/react'
import { UserProvider } from '@auth0/nextjs-auth0/client';

export default function Providers({children}) {
  return (
    <UserProvider>
      <NextUIProvider>
        {children}
      </NextUIProvider>
    </UserProvider>
  )
}