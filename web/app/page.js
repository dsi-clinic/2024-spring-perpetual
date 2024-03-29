import Image from "next/image";

export default function Home() {
  return (
    <>
      <div className="z-10 w-full max-w-xl px-5 xl:px-0 flex flex-col justify-center items-center">
        <h1
          className="animate-fade-up bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display text-4xl font-bold tracking-[-0.02em] text-transparent drop-shadow-sm [text-wrap:balance] md:text-7xl md:leading-[5rem] p-1"
          style={{ animationDelay: "0.15s", animationFillMode: "forwards" }}
        >
          Reuse System Design Wizard
        </h1>
        <br/>
        <p
          className="mt-6 animate-fade-up text-center text-gray-500 [text-wrap:balance] md:text-xl"
          style={{ animationDelay: "0.25s", animationFillMode: "forwards" }}
        >
          Automated collection bin placement, routing optimization, and sensitivity analysis for any city.
        </p>
        <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-8 text-xl">
          <a href="/api/auth/login">Get Started</a>
        </button>
      </div>
    </>
  );
}
