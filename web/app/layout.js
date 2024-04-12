import { Inter, Pavanam } from "next/font/google";
import { Suspense } from "react";
import "./globals.css";
import Nav from '../components/nav';
import Providers from "./providers";


const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({ children }) {

  return (
    <html lang="en">
        <body className={inter.className}>
          <Providers>
            <Nav />
            <main className="flex flex-col min-h-screen w-full flex-col items-center justify-center py-32 px-8">
              {children}
            </main>
          </Providers>              
        </body>
    </html>
  );
}