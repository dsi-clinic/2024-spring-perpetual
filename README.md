# Perpetual

### Background

Perpetual is a non-profit organization that aims to reduce the use of single-use foodware. This repository contains a model that can help Perpetual simulate a centralized foodware reuse system in a city.

Perpetual provided data for the city of Galveston, one of its first partners for this project. This repository contains the dataset and an example on how to create a simulation of a centralized foodware reuse system for Galveston, TX. However, this repository can work for any city given that the relevant data is provided.

### Goal

A centralized reuse system consists of multiple articipating Foodware-service Entities (FSEs) which consists of all locations that provide food or drinks in disposable containers. Additionally, multiple collection bins are installed throughout the city to make it convenient for people to deposit reusable containers. These containers are then washed in a centralized washing facility and redistributed back to the participating FSEs.

Planning a feasible centralized reuse system comes with a lot of challenges. This repository uses a set of techniques to find the most optimal way of approaching this problem. The map below shows how a dataset of particiating locations and their expected volume of use can be turned into a map of optimal routes to serve the city of Galveston, Texas, in the most feasible manner.

![The interactive map can be found in the output directory](archive/mentor/images/galveston_map.png)
