export interface CalendarDate {
    year: number;
    month: number;
    day: number;
    hour: number;
    minute: number;
    second: number;
}

/**
 * Converts a Julian Day (JD) number into a Gregorian or Julian calendar date and time.
 *
 * Primary Sources:
 * 1. Meeus, Jean. "Astronomical Algorithms", 2nd Edition (1998), Willmann-Bell, Chapter 7, pp. 63–64.
 * 2. Fliegel, H. F., and Van Flandern, T. C. (1968). "A Machine Algorithm for Processing Calendar Dates".
 *    Communications of the ACM, Vol. 11, No. 10, p. 657.
 *
 * @param jd - The Julian Day number to convert.
 * @returns Object representing Year, Month (1-12), Day (1-31), Hour (0-23), Minute (0-59), Second (0-59).
 */
function jdToCalendar(jd: number): CalendarDate {
    const Z = Math.floor(jd + 0.5);
    const F = jd + 0.5 - Z;

    let A = Z;
    if (Z >= 2299161) {
        const alpha = Math.floor((Z - 1867216.25) / 36524.25);
        A = Z + 1 + alpha - Math.floor(alpha / 4);
    }
    const B = A + 1524;
    const C = Math.floor((B - 122.1) / 365.25);
    const D = Math.floor(365.25 * C);
    const E = Math.floor((B - D) / 30.6001);

    const dayFloat = B - D - Math.floor(30.6001 * E) + F;
    const month = E < 14 ? E - 1 : E - 13;
    const year = month > 2 ? C - 4716 : C - 4715;

    const day = Math.floor(dayFloat);
    const totalSeconds = Math.round((dayFloat - day) * 86400);
    const hour = Math.floor(totalSeconds / 3600);
    const minute = Math.floor((totalSeconds % 3600) / 60);
    const second = totalSeconds % 60;

    return { year, month, day, hour, minute, second };
}

function pad(n: number, width = 2): string {
    return n.toString().padStart(width, "0");
}

export function timestampTag(jd1: number, jd2: number): string {
    const { year, month, day, hour, minute, second } = jdToCalendar(jd1 + jd2);
    return `${year}${pad(month)}${pad(day)}_${pad(hour)}${pad(minute)}${pad(second)}`;
}
