import { getSession } from '@auth0/nextjs-auth0';

export default async function Dashboard() {

    const { user } = await getSession();

    return (
      user && <>
        <div className="z-10 w-full max-w-xl px-5 xl:px-0">
          <h1
            className="animate-fade-up bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display text-4xl font-bold tracking-[-0.02em] text-transparent drop-shadow-sm [text-wrap:balance] md:text-7xl md:leading-[5rem] p-1"
            style={{ animationDelay: "0.15s", animationFillMode: "forwards" }}
          >
            Welcome, {user.given_name}!
          </h1>
        </div>
        <div>
            Table
        </div>
      </>
    );
  }