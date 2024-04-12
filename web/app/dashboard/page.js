"use client";

import { useUser } from '@auth0/nextjs-auth0/client'
import Image from 'next/image';
import Link from 'next/link';

export default function Dashboard() {

    const { user, error, isLoading } = useUser();

    return (
      user && <>
        <div className="z-10 w-full max-w-xl px-5 xl:px-0">
          <h1
            className="animate-fade-up bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display text-3xl font-bold tracking-[-0.02em] text-transparent drop-shadow-sm [text-wrap:balance] md:text-7xl md:leading-[5rem] p-1"
            style={{ animationDelay: "0.15s", animationFillMode: "forwards" }}
          >
            Welcome, {user.given_name}!
          </h1>
        </div>
        <div className="py-4">
          Vivamus dapibus aliquam magna quis rhoncus. Vivamus viverra 
          risus dolor, eget vehicula diam mattis eu. Donec eget interdum 
          justo. Nam ut interdum elit. Phasellus porttitor congue erat, 
          non cursus mauris varius cursus. Pellentesque at nulla eu orci 
          tempor vestibulum. Nullam blandit nisl convallis nisl auctor 
          vehicula. Quisque tempor magna id nibh bibendum porta. Nam 
          tristique leo risus, efficitur sagittis diam lobortis quis. 
          Etiam pretium est sed viverra placerat.
        </div>
        <h2
          className="from-black to-stone-500 text-3xl font-bold tracking-[-0.02em] pb-4"
        >
          My Projects
        </h2>
        <div className="flex flex-col items-center justify-center bg-gray-50 w-full py-10">
          <Image
            src="/magnifying_glass.png"
            width={100}
            height={100}
            alt="Magnifying Glass"
          />
          <p className="text-center text-lg mt-4 text-gray-800">
            No projects found. Create one{" "} 
            <a 
              className="font-medium text-blue-600 dark:text-blue-500 hover:underline"
              href="/map"
            >
              here
            </a>.
          </p>
        </div>
      </>
    );
  }