#pragma once

#include <string>
#include <map>
#include "json.hpp"

class Settings
{
public:
    // Singleton pattern
    static Settings &getInstance() {
        static Settings instance;
        return instance;
    }
    Settings(Settings const&) = delete;
    void operator=(Settings const&) = delete;

    // Getters
    std::string getPlatformName() { return platformName; }
    std::string getDeviceName() { return deviceName; }
    std::map<unsigned int, std::string> getShortcuts() { return shortcuts; }
    int getWindowWidth() { return windowWidth; };
    int getWindowHeight() { return windowHeight; };
    float getRenderScale() { return renderScale; };
    bool getUseBitstack() { return clUseBitstack; }
    bool getUseSoA() { return clUseSoA; }

private:
    Settings();
    void init();
    void load();
    void import(nlohmann::json j);

    // Contents of settings singleton
    std::string platformName;
    std::string deviceName;
    std::map<unsigned int, std::string> shortcuts;
    bool clUseBitstack;
    bool clUseSoA;
    int windowWidth;
    int windowHeight;
    float renderScale;
};
