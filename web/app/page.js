"use client";

/**
 * Landing page for unauthenticated users.
 * @module app/page
 */

export default function LandingPage() {
  return (
    <main className="flex flex-col justify-center items-center min-h-screen">
      <div className="flex flex-col flex-grow w-full justify-center items-center bg-gradient-to-br from-indigo-50 via-white to-cyan-100">
        <h1
          className="max-w-xl animate-fade-up bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display text-4xl font-bold tracking-[-0.02em] text-transparent drop-shadow-sm [text-wrap:balance] md:text-7xl md:leading-[5rem] p-1"
          style={{ animationDelay: "0.15s", animationFillMode: "forwards" }}
        >
          Reuse System Design Wizard
        </h1>
        <br />
        <p
          className="max-w-xl mt-6 animate-fade-up text-center text-gray-500 [text-wrap:balance] md:text-xl"
          style={{ animationDelay: "0.25s", animationFillMode: "forwards" }}
        >
          Automated collection bin placement, routing optimization, and sensitivity analysis for any city.
        </p>
        <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-8 text-xl">
          <a href="/api/auth/login">Get Started</a>
        </button>
      </div>
    </main>
  );
}
