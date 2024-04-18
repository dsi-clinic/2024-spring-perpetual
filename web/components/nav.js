"use client";

import { useUser } from "@auth0/nextjs-auth0/client";
import {
  Avatar,
  Dropdown,
  DropdownItem,
  DropdownMenu,
  DropdownTrigger,
  Link,
  Navbar,
  NavbarBrand,
  NavbarContent,
  NavbarItem,
} from "@nextui-org/react";

export default function Nav() {
  const { user, error, isLoading } = useUser();

  return (
    <Navbar className="bg-gray-100 font-semibold" shouldHideOnScroll>
      <NavbarBrand>
        <Link href="/projects" aria-current="page">
          <img src="/perpetual_logo.png" alt="Logo" className="h-8 mr-2" />
        </Link>
      </NavbarBrand>
      <NavbarContent justify="right">
        <NavbarItem>
          <Link color="primary" href="#" aria-current="page">
            Dashboard
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Link color="primary" href="#" aria-current="page">
            Settings
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Link color="primary" href="#" aria-current="page">
            About
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Dropdown>
            <DropdownTrigger>
              <Avatar
                isBordered
                as="button"
                className="transition-transform"
                color="primary"
                name={isLoading ? "" : user.name}
                size="sm"
                src={isLoading ? "" : user.picture}
              />
            </DropdownTrigger>
            <DropdownMenu aria-label="Profile Actions" variant="flat">
              <DropdownItem key="profile" className="h-14 gap-2">
                <p className="font-semibold">Signed in as</p>
                <p className="font-semibold">{user?.email}</p>
              </DropdownItem>
              <DropdownItem key="logout" color="secondary">
                <a href="/api/auth/logout">Logout</a>
              </DropdownItem>
            </DropdownMenu>
          </Dropdown>
        </NavbarItem>
      </NavbarContent>
    </Navbar>
  );
}
