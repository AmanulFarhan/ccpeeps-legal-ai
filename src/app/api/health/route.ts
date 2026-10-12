import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase"

export async function GET() {
    const { error } = await supabase
    .from("legal_documents")
    .select("id")
    .limit(1);

    if (error) {
        console.log("Supabase health check failed: ", error.message);

        return NextResponse.json({
            status: "error",
            message: "Database query failed"
        }, {
            status: 500
        });
    }

    return NextResponse.json({
        status: "ok",
        message: "Supabase connection successfull"
    });
}
