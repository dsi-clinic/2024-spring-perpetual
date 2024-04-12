import Image from 'next/image';
import { getSession } from '@auth0/nextjs-auth0';

export default async function Navbar() {

  const { user } = await getSession();

  return (
    <header className="bg-gray-100 text-black p-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center">
          <img src="/perpetual_logo.png" alt="Logo" className="h-8 mr-2" />
        </div>
        <nav className="flex items-center">
          <a href="#home" className="px-4 py-2 hover:text-violet-800 hover:font-semibold">Home</a>
          <a href="#about" className="px-4 py-2 hover:text-violet-800 hover:font-semibold">About</a>
          <a href="#contact" className="px-4 py-2 hover:text-violet-800 hover:font-semibold">Contact</a>
          <button data-dropdown-toggle="dropdownHover" data-dropdown-trigger="hover">
            <Image
              src={user?.picture ?? ""}
              width={30}
              height={30}
              alt="User Profile Picture"
            />
          </button>
          <div id="dropdownHover" className="z-10 hidden bg-white divide-y divide-gray-100 rounded-lg shadow w-44 dark:bg-gray-700">
            <a href="#profile" className="block px-4 py-2 hover:bg-gray-100">Profile</a>
            <a href="#settings" className="block px-4 py-2 hover:bg-gray-100">Settings</a>
            <a href="#logout" className="block px-4 py-2 hover:bg-gray-100">Logout</a>
          </div>
        </nav>
      </div>
    </header>
  );
};