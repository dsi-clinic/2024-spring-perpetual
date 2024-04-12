"use client";

import {
  Avatar,
  Navbar,
  NavbarBrand, 
  NavbarContent, 
  NavbarItem, 
  Link, 
  DropdownItem,
  DropdownTrigger, 
  Dropdown,
  DropdownMenu,
  Spacer
} from "@nextui-org/react";
import { useUser } from '@auth0/nextjs-auth0/client';


export default function Nav() {

    const { user, error, isLoading } = useUser();
  
    return (
        <Navbar className="bg-gray-100 font-semibold" shouldHideOnScroll>
          <NavbarBrand>
            <img src="/perpetual_logo.png" alt="Logo" className="h-8 mr-2" />
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
              </Dropdown>
            </NavbarItem>
          </NavbarContent>
            {/* <NavbarContent className="hidden sm:flex gap-4" justify="end">
              <Spacer />
              <NavbarItem>
                <Link color="secondary" href="#" aria-current="page">
                  Home
                </Link>
                <NavbarItem>
                <Link color="foreground" href="#">
                  About
                </Link>
              </NavbarItem>
              </NavbarItem>
              <Dropdown placement="bottom-end">
                <DropdownTrigger>
                  <Avatar
                    isBordered
                    as="button"
                    className="transition-transform"
                    color="secondary"
                    name={isLoading ? "" : user.name}
                    size="sm"
                    src={isLoading ? "" : user.picture}
                  />
                </DropdownTrigger>
                <DropdownMenu aria-label="Profile Actions" variant="flat">
                  <DropdownItem key="profile" className="h-14 gap-2">
                    <p className="font-semibold">Signed in as</p>
                    <p className="font-semibold">zoey@example.com</p>
                  </DropdownItem>
                  <DropdownItem key="settings">My Settings</DropdownItem>
                  <DropdownItem key="team_settings">Team Settings</DropdownItem>
                  <DropdownItem key="analytics">Analytics</DropdownItem>
                  <DropdownItem key="system">System</DropdownItem>
                  <DropdownItem key="configurations">Configurations</DropdownItem>
                  <DropdownItem key="help_and_feedback">Help & Feedback</DropdownItem>
                  <DropdownItem key="logout" color="danger">
                    Log Out
                  </DropdownItem>
                </DropdownMenu>
              </Dropdown>
            </NavbarContent> */}
        </Navbar>
    );
}
